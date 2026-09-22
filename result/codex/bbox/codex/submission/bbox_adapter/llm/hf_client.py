"""Local HuggingFace black-box LLM (Mixtral-8x7B-v0.1).

The model is only used through its text output: BBOX-ADAPTER never inspects its
logits or parameters, exactly like for the API models.  LoRA supervised
fine-tuning of the same checkpoint (the upper bound baseline of Table 2/6) is
implemented in :mod:`bbox_adapter.baselines.sft_lora`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .base import BlackBoxLLM, Generation, LLMResult
from .cache import ResponseCache


class HuggingFaceClient(BlackBoxLLM):
    def __init__(
        self,
        name: str = "mistralai/Mixtral-8x7B-v0.1",
        dtype: str = "float16",
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> None:
        super().__init__(name=name, max_new_tokens=max_new_tokens)
        self.dtype = dtype
        self.device_map = device_map
        self.cache = ResponseCache(cache_dir)
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return self._model, self._tokenizer
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        dtype = getattr(torch, self.dtype, torch.float16)
        self._tokenizer = AutoTokenizer.from_pretrained(self.name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=dtype,
            device_map=self.device_map,
        )
        self._model.eval()
        return self._model, self._tokenizer

    def _generate(self, prompts, n, temperature, top_p, max_new_tokens, stop,
                  system_prompt, seed) -> List[LLMResult]:
        import torch

        model, tokenizer = self._lazy_load()
        results: List[LLMResult] = []
        for prompt in prompts:
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
            params = {
                "n": n,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,
            }
            cache_key = self.cache.key(self.name, prompt, params)
            cached = self.cache.get(cache_key)
            if cached is not None:
                results.append(
                    LLMResult(prompt=prompt, generations=[Generation(**g) for g in cached])
                )
                continue
            if seed is not None:
                torch.manual_seed(seed)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            do_sample = temperature > 0
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=max(temperature, 1e-5) if do_sample else None,
                    top_p=top_p if do_sample else None,
                    num_return_sequences=n,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prompt_length = inputs["input_ids"].shape[1]
            generations: List[Generation] = []
            for sequence in output:
                text = tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
                if stop:
                    for marker in stop:
                        index = text.find(marker)
                        if index != -1:
                            text = text[:index]
                generations.append(
                    Generation(
                        text=text,
                        prompt_tokens=int(prompt_length),
                        completion_tokens=int(sequence.shape[0] - prompt_length),
                    )
                )
            self.cache.put(cache_key, [g.__dict__ for g in generations])
            results.append(LLMResult(prompt=prompt, generations=generations))
        return results
