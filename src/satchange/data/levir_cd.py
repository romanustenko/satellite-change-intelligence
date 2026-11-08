"""Download, extract, locate, and validate the LEVIR-CD dataset."""

import shutil
from pathlib import Path
from typing import Any

import gdown
import numpy as np
import torch
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from satchange.data.second_binary import SECONDBinaryDataset




IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}

EXPECTED_SPLITS = (
    "train",
    "val",
    "test",
)




def download_levir_cd(
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

    if (
        not downloaded_files
        and not any(download_directory.iterdir())
    ):
        raise RuntimeError("LEVIR-CD download failed.")




def is_levir_cd_split(directory: Path) -> bool:
    """Check whether a directory contains A, B, and label folders."""

    return (
        (directory / "A").is_dir()
        and (directory / "B").is_dir()
        and (directory / "label").is_dir()
    )





def collect_image_names(directory: Path) -> set[str]:
    """Collect the names of all supported image files."""

    return {
        path.name
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    }




def index_levir_cd_split(
    split_directory: str | Path,
) -> tuple[
    list[str],
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
]:
    """Collect and validat aligned sample from one LEVIR-CD split."""

    split_directory = Path(split_directory)

    if not is_levir_cd_split(split_directory):
        raise RuntimeError(
            f"Invalid LEVIR-CD split structure: {split_directory}"
        )


    before_files = collect_image_files(
        split_directory / "A"
    )
    
    after_files = collect_image_files(
        split_directory / "B"
    )

    label_files = collect_image_files(
        split_directory / "label"
    )

    before_ids = set(before_files)
    after_ids = set(after_files)
    label_ids = set(label_files)


    if not before_ids:
        raise RuntimeError(
            f"No images were found in: {split_directory}"
        )

    if before_ids != after_ids:
        raise RuntimeError(
            "A and B sample identifiers do not match. "
            f"Missing in B: {len(before_ids - after_ids)}. "
            f"Extra in B: {len(after_ids - before_ids)}."
        )

    if before_ids != label_ids:
        raise RuntimeError(
            "Image and label identifiers do not match. "
            f"Missing labels: {len(before_ids - label_ids)}. "
            f"Extra labels: {len(label_ids - before_ids)}."
        )

    sample_ids = sorted(before_ids)

    return(
        sample_ids,
        before_files,
        after_files,
        label_files,
    )


def extract_levir_cd_archives(
    download_directory: Path,
    prepared_directory: Path,
) -> None:
    """Extract each LEVIR-CD archive into its own split directory."""

    archives = {
        archive.stem.lower(): archive
        for archive in download_directory.glob("*.zip")
    }

    missing_splits = [
        split_name
        for split_name in EXPECTED_SPLITS
        if split_name not in archives
    ]

    if missing_splits:
        raise FileNotFoundError(
            "Missing LEVIR-CD archives: "
            + ", ".join(missing_splits)
        )

    prepared_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name in EXPECTED_SPLITS:
        archive = archives[split_name]
        split_directory = prepared_directory / split_name

        if is_levir_cd_split(split_directory):
            print(
                f"Split is already extracted: {split_directory}\n"
                "Skipping extraction."
            )
            continue

        if split_directory.exists():
            shutil.rmtree(split_directory)

        split_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Extracting {archive.name} "
            f"into {split_directory}"
        )

        shutil.unpack_archive(
            filename=str(archive),
            extract_dir=str(split_directory),
        )




def collect_image_files(
    directory: Path,
) -> dict[str, Path]:
    """Index sipported image files by filename stem."""

    image_files: dict[str, Path] = {}

    for path in directory.iterdir():
        if (
            not path.is_file()
            or path.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue

        sample_id = path.stem

        if sample_id in image_files:
            raise RuntimeError(
                f"Duplicate sample identifier '{sample_id}' "
                f"in directory: {directory}"
            )
    
        image_files[sample_id] = path

    return image_files

    


def validate_levir_cd_split(
    split_directory: Path,
    split_name: str,
    ) -> int:
    """Validate one LEVIR-CD split and return its sample count."""

    (
        sample_ids,
        before_files,
        after_files,
        label_files,
    ) = index_levir_cd_split(split_directory)

        
    print()
    print(f"Split:         {split_name}")
    print(f"Before images: {len(before_files)}")
    print(f"After images:  {len(after_files)}")
    print(f"Labels:        {len(label_files)}")

    return len(sample_ids)




def validate_levir_cd(dataset_root: Path) -> None:
    """Validate all LEVIR-CD dataset splits."""

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    total_samples = 0

    for split_name in EXPECTED_SPLITS:
        split_directory = dataset_root / split_name

        sample_count = validate_levir_cd_split(
            split_directory=split_directory,
            split_name=split_name,
        )

        total_samples += sample_count

    print()
    print(f"Total samples: {total_samples}")
    print("LEVIR-CD structure is valid.")




def prepare_levir_cd(
    config: dict[str, Any],
    project_root: str | Path,
) -> Path:
    """Download, extract, and validate LEVIR-CD."""

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

    download_levir_cd(
        url=dataset_config["download_url"],
        download_directory=download_directory,
    )

    extract_levir_cd_archives(
        download_directory=download_directory,
        prepared_directory=prepared_directory,
    )

    validate_levir_cd(prepared_directory)

    return prepared_directory




class LEVIRCDDataset(Dataset):
    """Load paired LEVIR-CD images and binary change masks."""


    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_size: int | None,
        transform: Callable[
            [dict[str, torch.Tensor | str]],
            dict[str, torch.Tensor | str],
        ] | None = None,
    ) -> None:
        """Initialize one LEVIR-CD dataset split."""

        if split not in EXPECTED_SPLITS:
            raise ValueError(
                f"Unsupported split '{split}' "
                f"Expected one of: {EXPECTED_SPLITS}"
            )

        if image_size is not None and image_size <= 0:
            raise ValueError(
                f"Image size must be positive, received {image_size}"
            )

        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.split_directory = self.dataset_root / split
        self.transform = transform

        (
            self.sample_ids,
            self.before_files,
            self.after_files,
            self.label_files,
        ) = index_levir_cd_split(self.split_directory)


    def __len__(self) -> int:
        """Return the number of samples in the split."""

        return len(self.sample_ids)


    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | str]:
        """Load one before images, after image, and change mask."""

        sample_id = self.sample_ids[index]

        before = self._load_rgb_image(
            self.before_files[sample_id]
        )
        
        after = self._load_rgb_image(
            self.after_files[sample_id]
        )
        
        mask = self._load_binary_mask(
            self.label_files[sample_id]
        )

        sample: dict[str, torch.Tensor | str] = {
            "before": before,
            "after": after,
            "mask": mask,
            "sample_id": sample_id,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


    def _load_rgb_image(
        self,
        image_path: Path,
    ) -> torch.Tensor:
        """Load, resize, and normalize one RGB image."""
        
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.image_size is not None:
                image = image.resize(
                    (self.image_size, self.image_size),
                    resample=Image.Resampling.BILINEAR,
                )

            image_array = np.array(
                image,
                dtype=np.float32,
                copy=True,
            )

        image_array /= 255.0

        image_tensor = torch.from_numpy(
            image_array
        ).permute(2, 0, 1)

        return image_tensor.contiguous()


    def _load_binary_mask(
        self,
        mask_path: Path,
    ) -> torch.Tensor:
        """Load and binarize one change mask."""

        with Image.open(mask_path) as mask:
            mask = mask.convert("L")

            if self.image_size is not None:
                mask = mask.resize(
                    (self.image_size, self.image_size),
                    resample=Image.Resampling.NEAREST,
                )

            mask_array = np.array(
                mask,
                dtype=np.uint8,
                copy=True,
            )

        binary_mask = (
            mask_array > 0
        ).astype(np.float32)


        mask_tensor = torch.from_numpy(
            binary_mask
        ).unsqueeze(0)

        return mask_tensor.contiguous()


def create_dataset_dataloader(
    dataset_root: str | Path,
    split: str,
    image_size: int | None,
    batch_size: int,
    num_workers: int,
    transform: Callable[
        [dict[str, Any]],
        dict[str, Any],
    ] | None = None,
    shuffle: bool = False,
    pin_memory: bool = False,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    semantic_labels: bool = False,
) -> tuple["LEVIRCDDataset | SECONDBinaryDataset", DataLoader]:
    """Create one LEVIR-CD dataset split and its data loader."""

    if seed is not None and seed < 0:
        raise ValueError("seed cannot be negative.")

    # A SECOND root with a frozen split manifest is binary change detection with the
    # same sample contract (Research 2 WHERE stage); everything else is LEVIR-CD.
    from satchange.data.second_binary import SECONDBinaryDataset, is_second_binary_root

    dataset: LEVIRCDDataset | SECONDBinaryDataset
    if semantic_labels:
        from satchange.data.second_semantic import SECONDSemanticDataset

        if not is_second_binary_root(dataset_root):
            raise ValueError("semantic_labels requires a SECOND root with a split manifest.")
        dataset = SECONDSemanticDataset(
            dataset_root=dataset_root,
            split=split,
            image_size=image_size,
            transform=transform,
        )
    elif is_second_binary_root(dataset_root):
        dataset = SECONDBinaryDataset(
            dataset_root=dataset_root,
            split=split,
            image_size=image_size,
            transform=transform,
        )
    else:
        dataset = LEVIRCDDataset(
            dataset_root=dataset_root,
            split=split,
            image_size=image_size,
            transform=transform,
        )

    generator = None

    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
    )

    return dataset, data_loader