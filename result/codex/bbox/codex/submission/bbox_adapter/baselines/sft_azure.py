"""Azure OpenAI fine-tuning baseline (Table 2, Appendix F.2).

"For gpt-3.5-turbo, we use the OpenAI Fine-Tuning Service hosted on Azure [...]
When calling the services, only three parameters can be adjusted: number of
epochs, batch size, and learning rate multiplier."

The helper below converts the training split into the chat JSONL format,
uploads the file, launches the fine-tuning job (3 epochs by default, as in
Appendix F.2, and the grid of Table 9) and polls until the model is ready.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..data.loaders import QAExample
from ..data.prompts import (TRUTHFULQA_INSTRUCTION, build_generator_prompt)


@dataclass
class AzureSFTConfig:
    output_dir: str = "runs/azure_sft"
    num_epochs: int = 3            # Appendix F.2 (5 for the Table 9 best setting)
    batch_size: Optional[int] = None   # "auto" when None
    learning_rate_multiplier: Optional[float] = None
    suffix: str = "bbox-sft"
    poll_seconds: int = 60


def build_chat_records(dataset: str, examples: Sequence[QAExample]) -> List[Dict]:
    records = []
    for example in examples:
        system = TRUTHFULQA_INSTRUCTION if dataset == "truthfulqa" else None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": build_generator_prompt(dataset, example.question, example.choices)})
        messages.append({"role": "assistant", "content": example.solution or example.answer})
        records.append({"messages": messages})
    return records


def write_jsonl(records: Sequence[Dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


class AzureFineTuningREST:
    """Minimal REST client for the Azure OpenAI fine-tuning service.

    The paper fine-tunes gpt-3.5-turbo "through the API services" of Microsoft
    Azure; the REST endpoints below are the documented way of uploading the
    training file, creating a fine-tuning job and polling it.  Credentials come
    from the environment (``AZURE_OPENAI_ENDPOINT`` / ``AZURE_OPENAI_API_KEY`` /
    ``AZURE_OPENAI_API_VERSION``).
    """

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None,
                 api_version: Optional[str] = None, deploy_model: str = "gpt-3.5-turbo") -> None:
        self.endpoint = (endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self.api_version = (
            api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or "2023-12-01-preview"
        )
        self.deploy_model = deploy_model
        if not self.endpoint or not self.api_key:
            raise RuntimeError(
                "Azure credentials missing: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            )

    # ------------------------------------------------------------------ utils
    def _headers(self) -> Dict[str, str]:
        return {"api-key": self.api_key}

    def _url(self, path: str) -> str:
        return f"{self.endpoint}/openai/{path}?api-version={self.api_version}"

    # ------------------------------------------------------------------- api
    def upload_file(self, path: str, purpose: str = "fine-tune") -> str:
        import requests

        with open(path, "rb") as handle:
            response = requests.post(
                self._url("files"),
                headers=self._headers(),
                files={"file": (os.path.basename(path), handle, "application/json")},
                data={"purpose": purpose},
                timeout=600,
            )
        response.raise_for_status()
        return response.json()["id"]

    def create_job(self, training_file: str, hyperparameters: Dict[str, object],
                   suffix: str = "bbox-sft") -> str:
        import requests

        payload = {
            "model": self.deploy_model,
            "training_file": training_file,
            "hyperparameters": hyperparameters,
            "suffix": suffix,
        }
        response = requests.post(self._url("fine_tuning/jobs"), headers=self._headers(),
                                 json=payload, timeout=600)
        response.raise_for_status()
        return response.json()["id"]

    def get_job(self, job_id: str) -> Dict:
        import requests

        response = requests.get(self._url(f"fine_tuning/jobs/{job_id}"), headers=self._headers(),
                                timeout=120)
        response.raise_for_status()
        return response.json()


def run_azure_sft(dataset: str, examples: Sequence[QAExample], config: AzureSFTConfig,
                  client=None) -> str:
    """Launch a fine-tuning job and return the resulting model name."""

    os.makedirs(config.output_dir, exist_ok=True)
    train_path = write_jsonl(
        build_chat_records(dataset, examples),
        os.path.join(config.output_dir, "train.jsonl"),
    )

    hyperparameters: Dict[str, object] = {"n_epochs": config.num_epochs}
    if config.batch_size is not None:
        hyperparameters["batch_size"] = config.batch_size
    if config.learning_rate_multiplier is not None:
        hyperparameters["learning_rate_multiplier"] = config.learning_rate_multiplier

    if client is None:
        # Default: the Azure REST service described above.
        rest = AzureFineTuningREST()
        file_id = rest.upload_file(train_path)
        print(f"[azure-sft] uploaded {train_path} as {file_id}")
        job_id = rest.create_job(file_id, hyperparameters, config.suffix)
        print(f"[azure-sft] created job {job_id}")
        while True:
            job = rest.get_job(job_id)
            status = job.get("status")
            print(f"[azure-sft] status={status}")
            if status in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(config.poll_seconds)
        if job.get("status") != "succeeded":
            raise RuntimeError(f"Fine-tuning job {job_id} ended with status {job.get('status')}")
        model_name = job.get("fine_tuned_model")
        _write_result(config, job_id, model_name)
        print(f"[azure-sft] fine-tuned model: {model_name}")
        return model_name

    # Alternative: an already constructed openai SDK client.
    uploaded = client.files.create(file=open(train_path, "rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(training_file=uploaded.id, model="gpt-3.5-turbo",
                                         hyperparameters=hyperparameters, suffix=config.suffix)
    print(f"[azure-sft] created job {job.id}")
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"[azure-sft] status={job.status}")
        if job.status in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(config.poll_seconds)
    if job.status != "succeeded":
        raise RuntimeError(f"Fine-tuning job {job.id} ended with status {job.status}")
    print(f"[azure-sft] fine-tuned model: {job.fine_tuned_model}")
    _write_result(config, job.id, job.fine_tuned_model)
    return job.fine_tuned_model


def _write_result(config: AzureSFTConfig, job_id: str, model_name: Optional[str]) -> None:
    with open(os.path.join(config.output_dir, "job.json"), "w", encoding="utf-8") as handle:
        json.dump({"job_id": job_id, "model": model_name}, handle, indent=2)
