from pathlib import Path
import shutil
import kagglehub

from src.config import DATA_RAW_DIR, DATASET_SLUG


def download_dataset() -> Path:
    """Download the Kaggle dataset and copy CSV files into data/raw."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))

    csv_files = list(dataset_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset: {dataset_path}")

    for csv_file in csv_files:
        destination = DATA_RAW_DIR / csv_file.name
        shutil.copy2(csv_file, destination)
        print(f"Copied {csv_file.name} -> {destination}")

    print(f"Dataset ready in: {DATA_RAW_DIR}")
    return DATA_RAW_DIR


if __name__ == "__main__":
    download_dataset()
