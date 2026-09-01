"""Shared utility functions used across the project."""

from pathlib import Path
from typing import Any

# the dependence typing is necessary in order to allow every type in
# yaml to be used
import yaml


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary."""

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {config_path}"
        )

    if not config_path.is_file():
        raise ValueError(
            f"Configuration path is not a file: {config_path}"
        )

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {config_path}"
        )

    if not isinstance(config, dict):
        raise ValueError(
            "The root YAML element must be a mapping"
        )

    return config
