"""PPLM (Dathathri et al., 2019) -- used in Section 4.2 to generate the toxic
half of the preference pairs.

PPLM steers generation by taking gradient steps on the past key/value cache in
the direction that increases the log-likelihood of an attribute classifier::

    p(y | a) ~ p(y) p(a | y)

The attribute classifier in the paper is the toxicity probe itself
(``W_toxic``), which takes the *mean residual stream of the last layer* as input.
We implement the standard three-part PPLM step (perturbation of the past,
geometric-mean fusion of the perturbed and unperturbed distributions, and a
KL-divergence post-norm), with the hyperparameters of Table 9:

    step size 0.4, temperature 1, top-k 10, num iterations 50, window 0,
    horizon 1, decay False, gamma 1, gm-scale 0.95, kl-scale 0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from .architecture import TransformerInternals
from .utils import load_tokenizer


@dataclass
class PPLMConfig:
    step_size: float = 0.4
    temperature: float = 1.0
    top_k: int = 10
    num_iterations: int = 50
    window_length: int = 0
    horizon_length: int = 1
    decay: bool = False
    gamma: float = 1.0
    gm_scale: float = 0.95
    kl_scale: float = 0.1
    min_length: int = 20
    max_length: int = 40
    grad_length: int = 10000
    repetition_penalty: float = 1.0
    stop_on_eos: bool = True


class AttributeClassifier:
    """``p(a | x)`` where ``x`` is the mean-pooled last-layer residual stream.

    This is precisely the toxicity probe ``W_toxic`` of Section 3.1, i.e. the
    same classifier that PPLM uses as the attribute model in Section 4.2.
    """

    def __init__(self, probe, tokenizer=None, device: Optional[str] = None,
                 temperature: float = 1.0):
        self.probe = probe
        self.device = device or str(next(probe.parameters()).device)
        self.probe.to(self.device).eval()
        self.temperature = temperature
        self.tokenizer = tokenizer

    def log_prob(self, mean_hidden: torch.Tensor, class_idx: int = 1) -> torch.Tensor:
        logits = self.probe(mean_hidden.to(self.device))[:, class_idx]
        return logits / max(self.temperature, 1e-6)

    def prob(self, mean_hidden: torch.Tensor, class_idx: int = 1) -> torch.Tensor:
        return torch.sigmoid(self.log_prob(mean_hidden, class_idx))


class PPLMGenerator:
    """Attribute-controlled generation on top of a causal LM."""

    def __init__(self, model: torch.nn.Module, classifier: AttributeClassifier,
                 cfg: Optional[PPLMConfig] = None, device: Optional[str] = None,
                 tokenizer=None, class_idx: int = 1):
        self.model = model
        self.internals = TransformerInternals(model)
        self.classifier = classifier
        self.cfg = cfg or PPLMConfig()
        self.device = device or str(next(model.parameters()).device)
        self.tokenizer = tokenizer or load_tokenizer()
        self.class_idx = class_idx
        self.model.eval()

    # ------------------------------------------------------------------ utils
    def _last_layer_hidden(self, outputs_hidden, attention_mask) -> torch.Tensor:
        h = outputs_hidden[-1]
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def _encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def _forward_with_past(self, input_ids: torch.Tensor, past=None, perturb=None):
        """Run the model, optionally adding ``perturb`` to the past key/value cache."""
        if past is None:
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            past = self._extract_past(outputs)
        else:
            outputs = self.model(input_ids=input_ids, past_key_values=past, output_hidden_states=True)
        return outputs, past

    def _extract_past(self, outputs):
        return getattr(outputs, "past_key_values", None)

    # ------------------------------------------------------- PPLM core steps
    def _perturb_past(self, past, accumulated_sum: Optional[torch.Tensor], n_accumulated: int,
                      input_ids: torch.Tensor):
        """Take ``num_iterations`` gradient steps on the past cache (Algorithm 1)."""
        cfg = self.cfg
        if past is None or past[0][0] is None:
            return past
        # clone to a differentiable version
        perturbed = []
        for layer_past in past:
            perturbed.append(tuple(p.detach().clone().requires_grad_(True) for p in layer_past))
        perturbed = tuple(perturbed)
        grads = None
        for _ in range(cfg.num_iterations):
            outputs = self.model(input_ids=input_ids, past_key_values=perturbed,
                                 output_hidden_states=True, use_cache=True)
            hidden = outputs.hidden_states[-1]
            mean_hidden = self._running_mean(hidden, accumulated_sum, n_accumulated)
            loss = -self.classifier.log_prob(mean_hidden, self.class_idx).sum()
            grads = torch.autograd.grad(loss, [p for lp in perturbed for p in lp])
            # apply the gradient step
            new_past = []
            g = 0
            for lp in perturbed:
                new_layer = []
                for p in lp:
                    grad = grads[g]
                    g += 1
                    norm = grad.norm()
                    if norm > 1e-8:
                        step = cfg.step_size * grad / norm
                    else:
                        step = torch.zeros_like(grad)
                    new_layer.append((p - step).detach().clone().requires_grad_(True))
                new_past.append(tuple(new_layer))
            perturbed = tuple(new_past)
        return perturbed

    @staticmethod
    def _running_mean(hidden: torch.Tensor, accumulated_sum: Optional[torch.Tensor],
                      n_accumulated: int) -> torch.Tensor:
        """Mean of the last-layer residual stream over the whole prefix seen so far.

        The toxicity probe of Section 3.1 is trained on the *mean over all
        timesteps* of the last-layer residual stream, so PPLM steers that same
        quantity: the sum of the hidden states of all previously processed
        positions plus the hidden states of the current forward pass.
        """
        cur_sum = hidden.sum(dim=1)
        total_sum = cur_sum if accumulated_sum is None else accumulated_sum + cur_sum
        total_n = hidden.shape[1] if accumulated_sum is None else n_accumulated + hidden.shape[1]
        return total_sum / max(total_n, 1)

    # ------------------------------------------------------------- sampling
    def _top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)

    def _sample(self, logits: torch.Tensor, prev_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        logits = logits.clone()
        if cfg.repetition_penalty != 1.0 and prev_ids.numel() > 0:
            for tok in set(prev_ids.view(-1).tolist()):
                if logits[0, tok] < 0:
                    logits[0, tok] *= cfg.repetition_penalty
                else:
                    logits[0, tok] /= cfg.repetition_penalty
        logits = self._top_k_logits(logits, cfg.top_k)
        probs = F.softmax(logits / max(cfg.temperature, 1e-6), dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _post_norm(self, unpert_logits: torch.Tensor, pert_logits: torch.Tensor) -> torch.Tensor:
        """Geometric-mean fusion + (optional) KL-divergence post-norm."""
        cfg = self.cfg
        unpert_probs = F.softmax(unpert_logits, dim=-1)
        pert_probs = F.softmax(pert_logits, dim=-1)
        fused = (pert_probs ** cfg.gm_scale) * (unpert_probs ** (1 - cfg.gm_scale))
        if cfg.kl_scale > 0:
            # KL(pert || unpert) used as an entropy-like penalty, following the
            # reference implementation of PPLM.
            kl = (pert_probs * (torch.log(pert_probs + 1e-10) - torch.log(unpert_probs + 1e-10))).sum(dim=-1)
            fused = fused / (1e-10 + kl.unsqueeze(-1) * cfg.kl_scale)
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return torch.log(fused + 1e-10)

    @torch.no_grad()
    def generate(self, prompt: str, seed: Optional[int] = None) -> str:
        """Generate a toxic continuation of ``prompt`` (Algorithm 1 of PPLM)."""
        cfg = self.cfg
        if seed is not None:
            torch.manual_seed(seed)
        input_ids = self._encode(prompt)
        accumulated_sum: Optional[torch.Tensor] = None
        n_accumulated = 0
        past = None
        generated: List[int] = []
        for step in range(cfg.max_length):
            with torch.enable_grad():
                outputs = self.model(input_ids=input_ids, past_key_values=past,
                                     output_hidden_states=True, use_cache=True)
                hidden = outputs.hidden_states[-1]
                # update the running statistics with the positions seen in this step
                new_sum = hidden.detach().sum(dim=1)
                accumulated_sum = new_sum if accumulated_sum is None else accumulated_sum + new_sum
                n_accumulated += hidden.shape[1]
                unpert_logits = self.model.lm_head(hidden[:, -1]).detach()
                raw_past = outputs.past_key_values
                # gradient perturbation of the cache
                if raw_past is not None and self.internals.arch == "mlp":
                    perturbed_past = self._perturb_past(
                        raw_past, accumulated_sum - new_sum, n_accumulated - hidden.shape[1], input_ids)
                    outputs_p = self.model(input_ids=input_ids, past_key_values=perturbed_past,
                                           output_hidden_states=True, use_cache=True)
                    pert_logits = self.model.lm_head(outputs_p.hidden_states[-1][:, -1]).detach()
                    logits = self._post_norm(unpert_logits[0], pert_logits[0]).unsqueeze(0)
                    past = raw_past
                else:  # fall back to unperturbed sampling
                    logits = unpert_logits
                    past = raw_past
            prev = torch.tensor(generated, device=self.device) if generated else torch.tensor([], dtype=torch.long, device=self.device)
            next_id = self._sample(logits, prev)
            generated.append(int(next_id))
            input_ids = next_id
            if cfg.stop_on_eos and int(next_id) == self.tokenizer.eos_token_id and step >= cfg.min_length:
                break
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def build_toxic_continuation(model: torch.nn.Module, probe, prompt: str,
                             cfg: Optional[PPLMConfig] = None, tokenizer=None,
                             device: Optional[str] = None, seed: Optional[int] = None) -> str:
    """Convenience wrapper: PPLM generation steered towards the toxic class."""
    tok = tokenizer or load_tokenizer()
    classifier = AttributeClassifier(probe, tokenizer=tok, device=device)
    gen = PPLMGenerator(model, classifier, cfg=cfg, device=device, tokenizer=tok)
    return gen.generate(prompt, seed=seed)
