"""Agent-owned AutoDL configuration used by controls and the admin projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from apps.common.billing_constants import CREDITS_PER_CNY
from apps.v2.private_config import PrivateJsonConfiguration


DEFAULT_API_BASE_URL = "https://api.autodl.com"
DEFAULT_REGIONS = ("westDC3",)
DEFAULT_GPU_PRICE_UNITS = {
    "RTX 4090": 1800,
    "RTX 3090": 1200,
    "RTX 3080 Ti": 900,
    "RTX A5000": 1200,
    "RTX A4000": 700,
    "A40": 2200,
    "A100": 6000,
    "H100": 12000,
}
API_KEY_MASK = "***"


@dataclass(frozen=True, slots=True)
class AutoDLRuntimeConfig:
    api_key: str
    image_uuid: str
    api_base_url: str
    regions: tuple[str, ...]
    credits_per_cny: int
    autodl_price_units_per_cny: int
    cuda_v_from: int
    cuda_v_to: int
    memory_size_to: int
    cpu_num_to: int
    price_from: int
    price_to: int
    start_cmd: str
    cmd_before_shutdown: str
    system_disk_expand_gb: int
    provision_wait_seconds: int
    max_create_retries: int
    delete_deployment_on_finish: bool
    gpu_price_units: dict[str, int]


class AutoDLConfigurationAuthority:
    def __init__(
        self,
        path: Path,
        *,
        fallback_api_key: str = "",
        fallback_base_url: str = DEFAULT_API_BASE_URL,
    ) -> None:
        self.store = PrivateJsonConfiguration(
            path,
            schema_version="deepevol-autodl-config@v2",
        )
        self.fallback_api_key = fallback_api_key.strip()
        self.fallback_base_url = fallback_base_url.strip() or DEFAULT_API_BASE_URL
        _http_origin(self.fallback_base_url)

    @property
    def path(self) -> Path:
        return self.store.path

    def public(self) -> dict[str, Any]:
        config = self.runtime()
        has_api_key = bool(config.api_key)
        return {
            "configured": has_api_key and bool(config.image_uuid),
            "has_api_key": has_api_key,
            "api_key_masked": API_KEY_MASK if has_api_key else "",
            "image_uuid": config.image_uuid,
            "api_base_url": config.api_base_url,
            "regions": list(config.regions),
            "credits_per_cny": config.credits_per_cny,
            "autodl_price_units_per_cny": config.autodl_price_units_per_cny,
            "cuda_v_from": config.cuda_v_from,
            "cuda_v_to": config.cuda_v_to,
            "memory_size_to": config.memory_size_to,
            "cpu_num_to": config.cpu_num_to,
            "price_from": config.price_from,
            "price_to": config.price_to,
            "start_cmd": config.start_cmd,
            "cmd_before_shutdown": config.cmd_before_shutdown,
            "system_disk_expand_gb": config.system_disk_expand_gb,
            "provision_wait_seconds": config.provision_wait_seconds,
            "max_create_retries": config.max_create_retries,
            "delete_deployment_on_finish": config.delete_deployment_on_finish,
            "gpu_price_units": dict(config.gpu_price_units),
            "config_file": str(self.path),
        }

    def update(self, values: Mapping[str, Any]) -> dict[str, Any]:
        current = self.runtime()
        supplied_key = _text(values.get("api_key"))
        if supplied_key == API_KEY_MASK:
            supplied_key = ""
        merged = {**asdict(current), **dict(values)}
        merged["api_key"] = supplied_key or current.api_key
        normalized = _normalize(merged)
        if normalized.api_key and not normalized.image_uuid:
            raise ValueError(
                "配置了 AutoDL API Key 时必须同时填写镜像 image_uuid，否则无法创建部署"
            )
        self.store.write(asdict(normalized))
        return self.public()

    def runtime(self) -> AutoDLRuntimeConfig:
        persisted = self.store.read() or {}
        normalized = _normalize(persisted)
        return replace(
            normalized,
            api_key=normalized.api_key or self.fallback_api_key,
            api_base_url=(
                normalized.api_base_url
                if _text(persisted.get("api_base_url"))
                else self.fallback_base_url.rstrip("/")
            ),
        )

    def control_settings(self) -> tuple[str, str]:
        config = self.runtime()
        api_key = config.api_key
        if not api_key:
            raise RuntimeError("AutoDL API key is not configured")
        return api_key, config.api_base_url


def _normalize(raw: Mapping[str, Any]) -> AutoDLRuntimeConfig:
    base_url = _text(raw.get("api_base_url")) or DEFAULT_API_BASE_URL
    _http_origin(base_url)
    regions = _regions(raw.get("regions")) or list(DEFAULT_REGIONS)
    gpu_prices = _gpu_prices(raw.get("gpu_price_units"))
    return AutoDLRuntimeConfig(
        api_key=_text(raw.get("api_key")),
        image_uuid=_text(raw.get("image_uuid")),
        api_base_url=base_url.rstrip("/"),
        regions=tuple(regions),
        credits_per_cny=_positive(raw.get("credits_per_cny"), CREDITS_PER_CNY),
        autodl_price_units_per_cny=_positive(
            raw.get("autodl_price_units_per_cny"), 1000
        ),
        cuda_v_from=_positive(raw.get("cuda_v_from"), 113),
        cuda_v_to=_positive(raw.get("cuda_v_to"), 128),
        memory_size_to=_positive(raw.get("memory_size_to"), 256),
        cpu_num_to=_positive(raw.get("cpu_num_to"), 100),
        price_from=_non_negative(raw.get("price_from"), 10),
        price_to=_positive(raw.get("price_to"), 9000),
        start_cmd=_text(raw.get("start_cmd")) or "sleep 2147483647",
        cmd_before_shutdown=_text(raw.get("cmd_before_shutdown")),
        system_disk_expand_gb=_non_negative(raw.get("system_disk_expand_gb"), 0),
        provision_wait_seconds=_non_negative(raw.get("provision_wait_seconds"), 60),
        max_create_retries=_non_negative(raw.get("max_create_retries"), 5),
        delete_deployment_on_finish=_boolean(
            raw.get("delete_deployment_on_finish"), True
        ),
        gpu_price_units=gpu_prices,
    )


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _regions(value: Any) -> list[str]:
    if isinstance(value, str):
        values = value.replace("\n", ",").split(",")
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _positive(value: Any, default: int) -> int:
    parsed = _integer(value, default)
    return parsed if parsed > 0 else default


def _non_negative(value: Any, default: int) -> int:
    parsed = _integer(value, default)
    return parsed if parsed >= 0 else default


def _integer(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _boolean(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"1", "true", "yes", "on"}:
            return True
        if value.strip().lower() in {"0", "false", "no", "off"}:
            return False
    return default


def _gpu_prices(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return dict(DEFAULT_GPU_PRICE_UNITS)
    result = {
        str(key).strip(): _non_negative(item, 0)
        for key, item in value.items()
        if str(key).strip()
    }
    return result or dict(DEFAULT_GPU_PRICE_UNITS)


def _http_origin(value: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("AutoDL API base URL must be an absolute HTTP(S) URL")


__all__ = [
    "API_KEY_MASK",
    "DEFAULT_API_BASE_URL",
    "DEFAULT_GPU_PRICE_UNITS",
    "DEFAULT_REGIONS",
    "AutoDLConfigurationAuthority",
    "AutoDLRuntimeConfig",
]
