import zipfile
from pathlib import Path


#project root
BASE_DIR = Path(__file__).resolve().parent

#file zip path
ZIP_PATH = BASE_DIR / "archive.zip"

#dataset destination fold
DATASET_DIR = BASE_DIR / "dataset"


def extract_dataset():
    """
    Extracts the dataset from archive.zip if not already extracted.
    """

    if DATASET_DIR.exists():
        print("Dataset already extracted. Skipping extraction")
        return

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Archive not found at {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    print("Dataset extracted successfully")


if __name__ == "__main__":
    extract_dataset()