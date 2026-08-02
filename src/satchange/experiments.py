"""Experiment configuration selection and valudation."""


from pathlib import Path
from typing import Any

from satchange.utils import load_config


FC_SIAM_DIFF_CONFIG_FILES = {
    "full": "fc_siam_diff_full.yaml",
    "resized": "fc_siam_diff_resized.yaml",
}


def load_fc_siam_diff_experiment(
    experiment: str,
    project_root: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Load and validate one FC-Siam-Diff experiment configuration."""

    experiment = experiment.strip().lower()

    if experiment not in FC_SIAM_DIFF_CONFIG_FILES:
        raise ValueError(
            f"Unknown FC-SIAM-Diff experiment: '{experiment}' .",
            f"Expected one of: {tuple(FC_SIAM_DIFF_CONFIG_FILES)}"
        )


    project_root = Path(project_root).resolve()

    config_path = (
        project_root
        / "configs"
        / FC_SIAM_DIFF_CONFIG_FILES[experiment]
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Experiment configuration was not found: {config_path}"
        )

    config = load_config(config_path)

    _validate_fc_siam_diff_experiment(
        config=config,
        experiment=experiment,
    )

    return config, config_path


def _validate_fc_siam_diff_experiment(
    experiment: str,
    config: dict[str, Any],
) -> None:
    """Check that the configuration matches the selected experiment."""

    data_config = config.get("data")

    if not isinstance(data_config, dict):
        raise ValueError(
            f"The configuration must contain a 'data' section."
        )

    image_size = data_config.get("image_size")
    batch_size = data_config.get("batch_size")

    if experiment == "full" and image_size is not None:
        raise ValueError(
            "The full-image experiment must use "
            "'data.image_size: null'."
        )

    if experiment == "resized":
        if (
            not isinstance(image_size, (int, float))
            or isinstance(image_size, bool)
            or image_size <= 0
        ):
            raise ValueError(
                "The resized experiment must use a positive integer "
                "for 'data.image_size'."
            )

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError(
            "'data.batch_size' must be a positive integer."
        )