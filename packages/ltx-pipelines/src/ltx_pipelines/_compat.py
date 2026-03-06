from pathlib import Path

DEFAULT_DISTILLED_LORA_FILENAME = "ltx-2.3-22b-distilled-lora-384.safetensors"
LEGACY_DISTILLED_LORA_FILENAME = "ltx-2-19b-distilled-lora-384.safetensors"


def has_lora_filename_pattern(path: str) -> bool:
    filename = Path(path).name
    stem, dot, suffix = filename.rpartition(".")
    if dot == "":
        return False
    return suffix == "safetensors" and "lora" in stem.lower()
