"""Download, extract, locate, and validate the LEVIR-CD dataset."""

import shutil
from pathlib import Path
from typing import Any

import gdown


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}

SPLIT_NAMES = {
    "train",
    "val",
    "test",
}


def download_dataset(
    url: str,
    download_directory: Path,
) -> None:
    """Download the public Google Drive folder."""

    if (
        download_directory.exists()
        and any(download_directory.iterdir())
    ):
        print(
            f"Download directory is not empty: {download_directory}\n"
            "Skipping download."
        )
        return

    download_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading LEVIR-CD...")

    downloaded_files = gdown.download_folder(
        url=url,
        output=str(download_directory),
        quiet=False,
    )

    if downloaded_files is None:
        raise RuntimeError("LEVIR-CD download failed.")


def extract_archives(
    download_directory: Path,
    prepared_directory: Path,
) -> None:
    """Extract every ZIP archive found in the download directory."""

    archives = list(
        download_directory.rglob("*.zip")
    )

    if not archives:
        print(
            "No ZIP archives were found. "
            "The dataset may already be extracted."
        )
        return

    prepared_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for archive in archives:
        print(f"Extracting: {archive}")

        shutil.unpack_archive(
            filename=str(archive),
            extract_dir=str(prepared_directory),
        )


def contains_image_triplet(directory: Path) -> bool:
    """Check whether a directory contains A, B, and label folders."""

    return (
        (directory / "A").is_dir()
        and (directory / "B").is_dir()
        and (directory / "label").is_dir()
    )


def contains_dataset_splits(directory: Path) -> bool:
    """Check whether a directory contains valid dataset split folders."""

    for split_name in SPLIT_NAMES:
        split_directory = directory / split_name

        if contains_image_triplet(split_directory):
            return True

    return False


def find_dataset_root(
    search_directories: list[Path],
) -> Path:
    """Find the root directory of the downloaded LEVIR-CD dataset."""

    for search_directory in search_directories:
        if not search_directory.exists():
            continue

        candidates = [
            search_directory,
            *[
                path
                for path in search_directory.rglob("*")
                if path.is_dir()
            ],
        ]

        for candidate in candidates:
            if contains_dataset_splits(candidate):
                return candidate

            if contains_image_triplet(candidate):
                return candidate

    raise FileNotFoundError(
        "Could not find a LEVIR-CD directory containing "
        "A, B, and label folders."
    )


def get_image_names(directory: Path) -> set[str]:
    """Return the names of all supported images in a directory."""

    return {
        path.stem
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    }


def validate_image_triplet(
    split_directory: Path,
    split_name: str,
) -> None:
    """Validate one A, B, and label directory triplet."""

    before_directory = split_directory / "A"
    after_directory = split_directory / "B"
    label_directory = split_directory / "label"

    before_names = get_image_names(before_directory)
    after_names = get_image_names(after_directory)
    label_names = get_image_names(label_directory)

    print()
    print(f"Split:         {split_name}")
    print(f"Directory:     {split_directory}")
    print(f"Before images: {len(before_names)}")
    print(f"After images:  {len(after_names)}")
    print(f"Labels:        {len(label_names)}")

    if not before_names:
        raise RuntimeError(
            f"The A directory contains no images: {before_directory}"
        )

    if before_names != after_names:
        missing_after = before_names - after_names
        extra_after = after_names - before_names

        raise RuntimeError(
            f"A and B filenames do not match in split '{split_name}'. "
            f"Missing in B: {len(missing_after)}. "
            f"Extra in B: {len(extra_after)}."
        )

    if before_names != label_names:
        missing_labels = before_names - label_names
        extra_labels = label_names - before_names

        raise RuntimeError(
            f"Image and label filenames do not match in split "
            f"'{split_name}'. "
            f"Missing labels: {len(missing_labels)}. "
            f"Extra labels: {len(extra_labels)}."
        )


def validate_dataset(dataset_root: Path) -> None:
    """Validate the LEVIR-CD directory structure and image filenames."""

    if contains_image_triplet(dataset_root):
        validate_image_triplet(
            split_directory=dataset_root,
            split_name="complete dataset",
        )

        print()
        print("LEVIR-CD structure is valid.")
        return

    validated_splits = 0

    for split_name in sorted(SPLIT_NAMES):
        split_directory = dataset_root / split_name

        if not contains_image_triplet(split_directory):
            continue

        validate_image_triplet(
            split_directory=split_directory,
            split_name=split_name,
        )

        validated_splits += 1

    if validated_splits == 0:
        raise RuntimeError(
            "No valid LEVIR-CD dataset splits were found."
        )

    print()
    print("LEVIR-CD structure is valid.")


def prepare_levir(
    config: dict[str, Any],
    project_root: str | Path,
) -> Path:
    """Run the complete LEVIR-CD preparation process."""

    project_root = Path(project_root)
    dataset_config = config["dataset"]

    download_directory = (
        project_root
        / dataset_config["download_directory"]
    )

    prepared_directory = (
        project_root
        / dataset_config["prepared_directory"]
    )

    download_dataset(
        url=dataset_config["download_url"],
        download_directory=download_directory,
    )

    extract_archives(
        download_directory=download_directory,
        prepared_directory=prepared_directory,
    )

    dataset_root = find_dataset_root(
        search_directories=[
            prepared_directory,
            download_directory,
        ],
    )

    validate_dataset(dataset_root)

    return dataset_root