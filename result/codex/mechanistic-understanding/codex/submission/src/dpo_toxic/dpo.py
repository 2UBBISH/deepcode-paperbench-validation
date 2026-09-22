"""Section 4.1 -- Direct Preference Optimization.

    L_DPO = -E[ log sigma( beta log P - beta log N ) ]
    P = pi_theta(y+ | w) / pi_ref(y+ | w),  N = pi_theta(y- | w) / pi_ref(y- | w)

Hyperparameters (Table 8 of the appendix, which the main text points to):

    learning rate 1e-6, batch size 4, RMSprop, gradient accumulation 1,
    max gradient norm 10, validation metric loss/valid, patience 10, beta 0.1.

The training data is split 90:10 (addendum) and the paper reports convergence
after approximately 6,700 preference pairs.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .utils import ensure_dir, save_json, set_seed


@dataclass
class DPOConfig:
    beta: float = 0.1
    learning_rate: float = 1e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 10.0
    optimizer: str = "rmsprop"
    epochs: int = 5
    max_length: int = 256
    val_fraction: float = 0.1
    val_every: int = 32           # in optimizer steps
    patience: int = 10            # validation-loss patience (paper: 10)
    seed: int = 0
    max_steps: Optional[int] = None   # for smoke tests
    max_train_pairs: Optional[int] = None
    log_every: int = 10
    device: Optional[str] = None
    label_smoothing: float = 0.0
    reference_free: bool = False


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def dpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
             ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor,
             beta: float = 0.1, reference_free: bool = False
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(loss, chosen_rewards, rejected_rewards)``."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    if reference_free:
        logits = pi_logratios
    else:
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_rewards, rejected_rewards


# --------------------------------------------------------------------------- #
# Log-probabilities
# --------------------------------------------------------------------------- #
def encode_pair(tokenizer, prompt: str, completion: str, max_length: int,
                device) -> Dict[str, torch.Tensor]:
    """Tokenize ``prompt + completion`` with the prompt tokens masked out.

    Only completion tokens contribute to ``log pi(y | w)``, matching the DPO
    objective which is defined over the continuation ``y`` given the prompt
    ``w``.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    comp_ids = tokenizer(completion, add_special_tokens=False).input_ids
    if len(comp_ids) == 0:
        comp_ids = [tokenizer.eos_token_id]
    ids = (prompt_ids + comp_ids)[:max_length]
    n_prompt = min(len(prompt_ids), len(ids) - 1)
    labels = [-100] * n_prompt + ids[n_prompt:]
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
    }


def sequence_logprob(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
    """Sum of log-probabilities of the (non-masked) label tokens."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = (targets != -100).float()
    logprobs = F.log_softmax(logits, dim=-1)
    gathered = logprobs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum()


def batch_logps(model, tokenizer, prompts: Sequence[str], completions: Sequence[str],
                max_length: int, device) -> torch.Tensor:
    out = []
    for p, c in zip(prompts, completions):
        enc = encode_pair(tokenizer, p, c, max_length, device)
        out.append(sequence_logprob(model, enc["input_ids"], enc["attention_mask"], enc["labels"]))
    return torch.stack(out)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def split_pairs(pairs: Sequence[Dict], val_fraction: float = 0.1,
                seed: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """90:10 train/validation split (addendum)."""
    n = len(pairs)
    n_val = max(1, int(round(val_fraction * n)))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = set(perm[:n_val])
    train = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [pairs[i] for i in perm[:n_val]]
    return train, val


def _optimizer(params, cfg: DPOConfig):
    if cfg.optimizer.lower() == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg.learning_rate)
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate)
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(params, lr=cfg.learning_rate)
    raise ValueError(cfg.optimizer)


@torch.no_grad()
def evaluate_dpo_loss(model, ref_model, tokenizer, pairs: Sequence[Dict],
                      cfg: DPOConfig, device) -> Dict[str, float]:
    model.eval()
    losses, accs = [], []
    for i in range(0, len(pairs), cfg.batch_size):
        batch = pairs[i: i + cfg.batch_size]
        prompts = [b["prompt"] for b in batch]
        chosen = [b["chosen"] for b in batch]
        rejected = [b["rejected"] for b in batch]
        pol_c = batch_logps(model, tokenizer, prompts, chosen, cfg.max_length, device)
        pol_r = batch_logps(model, tokenizer, prompts, rejected, cfg.max_length, device)
        with torch.no_grad():
            ref_c = batch_logps(ref_model, tokenizer, prompts, chosen, cfg.max_length, device)
            ref_r = batch_logps(ref_model, tokenizer, prompts, rejected, cfg.max_length, device)
        loss, c_rew, r_rew = dpo_loss(pol_c, pol_r, ref_c, ref_r, cfg.beta, cfg.reference_free)
        losses.append(float(loss.mean()))
        accs.append(float((c_rew > r_rew).float().mean()))
    model.train()
    return {"val_loss": sum(losses) / max(len(losses), 1),
            "val_accuracy": sum(accs) / max(len(accs), 1)}


def train_dpo(policy_model, ref_model, tokenizer, pairs: Sequence[Dict],
              cfg: Optional[DPOConfig] = None,
              out_dir: str = "artifacts/dpo",
              log_path: Optional[str] = None) -> Dict:
    """DPO training loop following Table 8 of the appendix."""
    cfg = cfg or DPOConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device) if cfg.device else next(policy_model.parameters()).device
    out = ensure_dir(out_dir)

    train_pairs, val_pairs = split_pairs(pairs, cfg.val_fraction, cfg.seed)
    if cfg.max_train_pairs:
        train_pairs = train_pairs[: cfg.max_train_pairs]

    policy_model.to(device)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    policy_model.train()

    opt = _optimizer([p for p in policy_model.parameters() if p.requires_grad], cfg)

    history: List[Dict] = []
    best_val, best_state, bad_epochs = float("inf"), None, 0
    global_step = 0
    n_pairs_seen = 0
    t0 = time.time()
    stop = False
    accum = cfg.gradient_accumulation_steps

    for epoch in range(cfg.epochs):
        g = torch.Generator().manual_seed(cfg.seed + epoch)
        order = torch.randperm(len(train_pairs), generator=g).tolist()
        for i in range(0, len(order), cfg.batch_size):
            batch = [train_pairs[j] for j in order[i: i + cfg.batch_size]]
            prompts = [b["prompt"] for b in batch]
            chosen = [b["chosen"] for b in batch]
            rejected = [b["rejected"] for b in batch]
            pol_c = batch_logps(policy_model, tokenizer, prompts, chosen, cfg.max_length, device)
            pol_r = batch_logps(policy_model, tokenizer, prompts, rejected, cfg.max_length, device)
            with torch.no_grad():
                ref_c = batch_logps(ref_model, tokenizer, prompts, chosen, cfg.max_length, device)
                ref_r = batch_logps(ref_model, tokenizer, prompts, rejected, cfg.max_length, device)
            loss, c_rew, r_rew = dpo_loss(pol_c, pol_r, ref_c, ref_r, cfg.beta, cfg.reference_free)
            (loss.mean() / accum).backward()
            n_pairs_seen += len(batch)

            if (global_step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if global_step % cfg.log_every == 0:
                rec = {"step": global_step, "epoch": epoch, "loss": float(loss.mean()),
                       "accuracy": float((c_rew > r_rew).float().mean()),
                       "reward_chosen": float(c_rew.mean()), "reward_rejected": float(r_rew.mean()),
                       "pairs_seen": n_pairs_seen, "elapsed": time.time() - t0}
                history.append(rec)
                print(f"[dpo] step {global_step} loss={rec['loss']:.4f} acc={rec['accuracy']:.3f} "
                      f"pairs={n_pairs_seen}")
                if log_path:
                    save_json(history, log_path)

            if val_pairs and (global_step + 1) % cfg.val_every == 0:
                val = evaluate_dpo_loss(policy_model, ref_model, tokenizer, val_pairs, cfg, device)
                val["step"] = global_step
                history.append({"validation": val})
                print(f"[dpo] validation @ step {global_step}: {val}")
                if log_path:
                    save_json(history, log_path)
                if val["val_loss"] < best_val - 1e-6:
                    best_val = val["val_loss"]
                    best_state = copy.deepcopy(policy_model.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg.patience:
                        print(f"[dpo] early stopping after {bad_epochs} non-improving validations")
                        stop = True
            global_step += 1
            if cfg.max_steps is not None and global_step >= cfg.max_steps:
                stop = True
            if stop:
                break
        if stop:
            break

    if best_state is not None:
        policy_model.load_state_dict(best_state)
    policy_model.save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "model")
    report = {"config": cfg.__dict__, "n_train_pairs": len(train_pairs),
              "n_val_pairs": len(val_pairs), "n_pairs_seen": n_pairs_seen,
              "best_val_loss": best_val if best_state is not None else None,
              "global_steps": global_step, "history": history}
    save_json(report, out / "train_report.json")
    return report


def build_reference_model(policy_model):
    """A frozen copy of the pre-trained model (``pi_ref``)."""
    ref = copy.deepcopy(policy_model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref
