"""Evaluation harness for the LVLM experiments of the paper.

Implemented here:

* Table 1 -- clean and adversarial (eps = 2/255, 4/255) performance of LLaVA-1.5
  and OpenFlamingo-9B with the different vision encoders on COCO, Flickr30k,
  TextVQA and VQAv2,
* Table 2 -- transfer attacks (adversarial COCO images crafted against one model
  and evaluated with another vision encoder / LVLM),
* Table 3 -- stealthy *targeted* attacks (10,000 iterations, see App. B.9),
* Table 5 -- POPE hallucination F1 score,
* Table 6 -- SQA-I accuracy,
* Table 14 -- clean and adversarial embedding losses (App. C.4).

The metric functions are the ones of :mod:`robust_clip.eval.cider`,
:mod:`robust_clip.eval.vqa` and the POPE / SQA helpers below.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch

from ..attacks.lvlm_attack import EnsembleAttackConfig, LVLMEnsembleAttack
from ..attacks.pgd import pgd_attack
from ..models.clip_encoder import CLIPEncoder
from ..models.lvlm import prompts as P
from ..models.lvlm.base import LVLM
from .cider import Cider
from .datasets import LVMLEvalSample, load_pope, pope_object
from .vqa import TextVQAAccuracy, VQAAccuracy

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# prompts and metrics per task
# ----------------------------------------------------------------------
def build_prompts(samples: Sequence[LVMLEvalSample], task: str, backend: str = "llava") -> List[str]:
    """Task specific prompts (LLaVA) or zero-shot context texts (OpenFlamingo)."""
    prompts: List[str] = []
    for sample in samples:
        if task in {"coco", "flickr30k"}:
            if backend == "llava":
                prompts.append(P.build_caption_prompt())
            else:
                prompts.append(P.OF_CAPTION_PROMPT)
        elif task in {"vqav2", "textvqa"}:
            if backend == "llava":
                prompts.append(P.build_vqa_prompt(sample.question))
            else:
                prompts.append(P.OF_VQA_PROMPT.format(question=sample.question))
        elif task == "pope":
            obj = sample.metadata.get("object") or pope_object(sample.question or "")
            if backend == "llava":
                prompts.append(P.build_pope_prompt(obj))
            else:
                prompts.append(P.OF_POPE_PROMPT.format(object=obj))
        elif task == "sqa":
            if backend == "llava":
                prompts.append(
                    P.build_sqa_prompt(
                        sample.question,
                        sample.metadata.get("choices", []),
                        context=sample.metadata.get("context", ""),
                    )
                )
            else:
                choices = ", ".join(sample.metadata.get("choices", []))
                prompts.append(f"<image>Question: {sample.question} Choices: {choices} Answer:")
        else:
            raise ValueError(f"unknown task {task!r}")
    return prompts


def make_metric(task: str, references: Sequence[Sequence[str]]) -> Callable[[Sequence[str], Sequence[Sequence[str]]], List[float]]:
    """Return the per-sample metric of the task (CIDEr or accuracy)."""
    if task in {"coco", "flickr30k"}:
        scorer = Cider().fit(list(references))
        return lambda generations, refs: scorer.score_batch(list(generations), list(refs))
    if task == "textvqa":
        metric = TextVQAAccuracy()
    else:  # vqav2, pope and sqa use the plain VQA accuracy / exact match
        metric = VQAAccuracy()
    if task == "pope":
        return lambda generations, refs: [
            float(parse_yes_no(gen) == parse_yes_no(ref[0])) for gen, ref in zip(generations, refs)
        ]
    if task == "sqa":
        return lambda generations, refs: [
            float(parse_sqa_answer(gen) == (ref[0].strip().lower())) for gen, ref in zip(generations, refs)
        ]
    return lambda generations, refs: metric.score_batch(list(generations), list(refs))


def parse_yes_no(text: str) -> Optional[str]:
    """Extract the yes/no answer of a POPE response (LLaVA answer convention)."""
    text = text.strip().lower()
    match = re.search(r"\b(yes|no)\b", text)
    if match:
        return match.group(1)
    return None


def parse_sqa_answer(text: str) -> str:
    """Extract the answer of an SQA response (letter, option text or free form)."""
    text = text.strip()
    for pattern in (r"[Tt]he answer is ([A-J])", r"^\s*([A-J])[\.\): ]", r"answer:\s*([A-J])\b"):
        match = re.search(pattern, text)
        if match:
            return match.group(1).lower()
    return text.split("\n")[0].strip().lower()


def pope_f1(predictions: Sequence[Optional[str]], labels: Sequence[str]) -> float:
    """F1 score of the POPE benchmark (``yes`` is the positive class)."""
    tp = fp = fn = 0
    for prediction, label in zip(predictions, labels):
        is_positive = parse_yes_no(label) == "yes"
        predicted_positive = parse_yes_no(prediction or "") == "yes"
        if predicted_positive and is_positive:
            tp += 1
        elif predicted_positive and not is_positive:
            fp += 1
        elif not predicted_positive and is_positive:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ----------------------------------------------------------------------
# Table 1: clean + adversarial LVLM evaluation
# ----------------------------------------------------------------------
def evaluate_clean(
    model: LVLM,
    samples: Sequence[LVMLEvalSample],
    task: str,
    batch_size: int = 8,
    max_new_tokens: int = 32,
    backend: str = "llava",
) -> Dict[str, float]:
    """Clean performance (all available samples, Sec. 4.1)."""
    prompts = build_prompts(samples, task, backend=backend)
    references = [sample.answers for sample in samples]
    metric = make_metric(task, references)
    generations: List[str] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        images = torch.stack([image_to_tensor(s.image) for s in batch])
        generations.extend(
            model.generate(images, prompts[start : start + batch_size], max_new_tokens=max_new_tokens)
        )

    if task == "pope":
        predictions = [parse_yes_no(text) for text in generations]
        labels = [sample.answers[0] for sample in samples]
        accuracy = sum(p == parse_yes_no(l) for p, l in zip(predictions, labels)) / max(len(samples), 1)
        return {"f1": 100.0 * pope_f1(predictions, labels), "accuracy": 100.0 * accuracy}

    if task == "sqa":
        correct = [
            float(parse_sqa_answer(text) == (sample.answers[0].strip().lower()))
            for text, sample in zip(generations, samples)
        ]
        return {"accuracy": 100.0 * sum(correct) / max(len(correct), 1)}

    scores = metric(generations, references)
    return {task_metric_name(task): 100.0 * sum(scores) / max(len(scores), 1)}


def task_metric_name(task: str) -> str:
    if task in {"coco", "flickr30k"}:
        return "cider"
    if task == "sqa":
        return "accuracy"
    return "accuracy"


def image_to_tensor(image, resolution: int = 224) -> torch.Tensor:
    """PIL image -> ``[3, H, W]`` tensor in ``[0, 1]`` (attack / model convention)."""
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    )
    return transform(image.convert("RGB"))


def evaluate_robust(
    model: LVLM,
    samples: Sequence[LVMLEvalSample],
    task: str,
    eps: Union[str, float] = "2/255",
    batch_size: int = 4,
    max_new_tokens: int = 32,
    backend: str = "llava",
    config: Optional[EnsembleAttackConfig] = None,
    save_adv_images: Optional[str] = None,
) -> Dict[str, object]:
    """Adversarial performance with the attack ensemble of App. B.6."""
    references = [sample.answers for sample in samples]
    prompts = build_prompts(samples, task, backend=backend)
    metric = make_metric(task, references)
    config = config or EnsembleAttackConfig(
        eps=str(eps),
        score_threshold=10.0 if task == "coco" else (2.0 if task == "flickr30k" else 0.0),
        targeted_use_word=(task != "textvqa"),
    )
    config.eps = str(eps)

    worst_scores: List[float] = []
    clean_scores: List[float] = []
    adversarial_images = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        images = torch.stack([image_to_tensor(s.image) for s in batch])
        attack = LVLMEnsembleAttack(model, config, metric_fn=metric)
        result = attack.run(
            images,
            prompts[start : start + batch_size],
            references[start : start + batch_size],
            is_vqa=task in {"vqav2", "textvqa"},
        )
        worst_scores.extend(result.scores)
        clean_scores.extend(result.clean_scores)
        adversarial_images.append(result.x_adv.cpu())

    if save_adv_images is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_adv_images)) or ".", exist_ok=True)
        torch.save(torch.cat(adversarial_images), save_adv_images)

    name = task_metric_name(task)
    return {
        f"clean_{name}": 100.0 * sum(clean_scores) / max(len(clean_scores), 1),
        f"robust_{name}": 100.0 * sum(worst_scores) / max(len(worst_scores), 1),
        "scores": worst_scores,
    }


# ----------------------------------------------------------------------
# Table 2: transfer attacks
# ----------------------------------------------------------------------
def transfer_attack(
    source_model: LVLM,
    target_model: LVLM,
    samples: Sequence[LVMLEvalSample],
    task: str = "coco",
    eps: Union[str, float] = "2/255",
    batch_size: int = 4,
    max_new_tokens: int = 32,
    source_backend: str = "llava",
    target_backend: str = "llava",
    adv_images_path: Optional[str] = None,
) -> Dict[str, float]:
    """Craft adversarial images with ``source_model`` and evaluate them with ``target_model``.

    The paper transfers the adversarial COCO images generated against OF-CLIP and
    LLaVA-CLIP to the respective other vision encoder / LVLM, restricting the
    evaluation to 200 samples.
    """
    references = [sample.answers for sample in samples]
    metric = make_metric(task, references)
    source_prompts = build_prompts(samples, task, backend=source_backend)
    target_prompts = build_prompts(samples, task, backend=target_backend)
    config = EnsembleAttackConfig(eps=str(eps))

    scores: List[float] = []
    adv_batches = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        images = torch.stack([image_to_tensor(s.image) for s in batch])
        attack = LVLMEnsembleAttack(source_model, config, metric_fn=metric)
        result = attack.run(
            images,
            source_prompts[start : start + batch_size],
            references[start : start + batch_size],
            is_vqa=task in {"vqav2", "textvqa"},
        )
        adv_batches.append(result.x_adv.cpu())
        generations = target_model.generate(
            result.x_adv, target_prompts[start : start + batch_size], max_new_tokens=max_new_tokens
        )
        scores.extend(metric(generations, references[start : start + batch_size]))

    if adv_images_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(adv_images_path)) or ".", exist_ok=True)
        torch.save(torch.cat(adv_batches), adv_images_path)
    name = task_metric_name(task)
    return {f"transfer_{name}": 100.0 * sum(scores) / max(len(scores), 1)}


# ----------------------------------------------------------------------
# Table 3: stealthy targeted attacks
# ----------------------------------------------------------------------
def targeted_attack(
    model: LVLM,
    image: torch.Tensor,
    target: str,
    eps: Union[str, float] = "4/255",
    iterations: int = 10000,
    alpha: str = "1/255",
    momentum: float = 0.9,
    prompt: Optional[str] = None,
    max_new_tokens: int = 32,
) -> Dict[str, object]:
    """Targeted l_inf attack that makes LLaVA output ``target`` (Sec. 4.2).

    The attack minimizes the teacher-forced NLL of the target string
    (App. B.9 uses 10,000 iterations for the main results and 500 for the
    ablation of Table 12).  It uses the PGD implementation of the addendum
    (normalized gradient with elementwise sign, momentum 0.9, uniform random
    initialization, l_inf ball around the non-normalized image); the
    *jailbreaking* attack of Qi et al. (2023) is the same PGD but without
    momentum (see :mod:`robust_clip.eval.jailbreak`).
    """
    prompt = prompt or P.build_caption_prompt()

    def forward_fn(x_adv):
        return x_adv

    def loss_fn(_, x_adv):
        return model.target_loss(x_adv, [prompt], [target], reduction="none")

    x_adv, final_loss = pgd_attack(
        image,
        forward_fn,
        loss_fn,
        eps=eps,
        alpha=alpha,
        steps=iterations,
        dtype=torch.float32,
        momentum=momentum,
        random_start=True,
        maximize=False,  # minimize the NLL of the target string
        return_best=True,
        track_best_every=50,
    )
    output = model.generate(x_adv, [prompt], max_new_tokens=max_new_tokens)[0]
    return {
        "generation": output,
        "success": target_success(output, target),
        "x_adv": x_adv,
        "loss": float(final_loss.mean()),
    }


def target_success(output: str, target: str) -> bool:
    """A targeted attack counts as successful if the output contains the target.

    Table 3 reports success rates and Fig. 3 shows the generated texts, so the
    criterion is the appearance of the attacker's target string in the answer
    (the paper additionally judges the quality of the outputs in Fig. 3).
    """
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    return normalize(target) in normalize(output)


def evaluate_targeted_attacks(
    model: LVLM,
    samples: Sequence[LVMLEvalSample],
    targets: Sequence[str] = tuple(P.TARGET_CAPTIONS),
    eps: Union[str, float] = "4/255",
    iterations: int = 10000,
    max_new_tokens: int = 32,
    output_json: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Run the targeted attacks of Table 3 for all target captions."""
    results: Dict[str, Dict[str, float]] = {}
    for target in targets:
        successes = 0
        for sample in samples:
            image = image_to_tensor(sample.image)[None]
            outcome = targeted_attack(
                model,
                image,
                target,
                eps=eps,
                iterations=iterations,
                max_new_tokens=max_new_tokens,
            )
            successes += int(outcome["success"])
        results[target] = {"success_rate": f"{successes}/{len(samples)}"}
    if output_json is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
        with open(output_json, "w") as handle:
            json.dump(results, handle, indent=2)
    return results


# ----------------------------------------------------------------------
# Table 5 / Table 6
# ----------------------------------------------------------------------
def evaluate_pope(
    model: LVLM,
    splits: Sequence[str] = ("random", "popular", "adversarial"),
    n_samples: Optional[int] = 3000,
    backend: str = "llava",
    **kwargs,
) -> Dict[str, float]:
    """POPE F1 score of all splits (Table 5)."""
    results = {}
    for split in splits:
        samples = load_pope(split=split, max_samples=n_samples)
        metrics = evaluate_clean(model, samples, "pope", backend=backend, **kwargs)
        results[split] = metrics["f1"]
    results["mean"] = sum(results.values()) / max(len(splits), 1)
    return results


def evaluate_sqa(
    model: LVLM,
    samples: Sequence[LVMLEvalSample],
    backend: str = "llava",
    **kwargs,
) -> Dict[str, float]:
    """SQA-I accuracy (Table 6)."""
    return evaluate_clean(model, samples, "sqa", backend=backend, **kwargs)


# ----------------------------------------------------------------------
# Table 14: embedding losses (App. C.4)
# ----------------------------------------------------------------------
def evaluate_embedding_loss(
    clip: CLIPEncoder,
    reference_clip: CLIPEncoder,
    loader,
    eps: Union[str, float] = "4/255",
    n_iter: int = 100,
    n_samples: int = 500,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """``E[L_clean]`` and ``E[L_adv]`` of Eqs. (4) and (5) on ImageNet (Table 14).

    A 100-step APGD with ``eps = 4/255`` maximizes the embedding loss
    ``||phi_FT(z) - phi_Org(x)||_2^2``.
    """
    from ..attacks.apgd import APGDAttack
    from ..training.losses import embedding_distance

    clean_values, adv_values = [], []
    seen = 0
    for images, _ in loader:
        if seen >= n_samples:
            break
        images = images[: n_samples - seen]
        with torch.no_grad():
            phi_org = reference_clip.image_embedding(images, normalize=False)
            phi_clean = clip.image_embedding(images, normalize=False)
        clean_values.extend(embedding_distance(phi_clean, phi_org).detach().cpu().tolist())

        def forward_fn(x_adv):
            return clip.image_embedding(x_adv, normalize=False)

        def loss_fn(phi_ft_adv, x_adv):
            return embedding_distance(phi_ft_adv, phi_org, norm="l2_squared")

        attacker = APGDAttack(eps=eps, n_iter=n_iter, loss="custom", dtype=torch.float32)
        x_adv = attacker.attack(images, forward_fn, custom_loss=loss_fn)
        with torch.no_grad():
            phi_adv = clip.image_embedding(x_adv, normalize=False)
        adv_values.extend(embedding_distance(phi_adv, phi_org).detach().cpu().tolist())
        seen += images.shape[0]

    return {
        "clean_embedding_loss": sum(clean_values) / max(len(clean_values), 1),
        "adversarial_embedding_loss": sum(adv_values) / max(len(adv_values), 1),
    }
