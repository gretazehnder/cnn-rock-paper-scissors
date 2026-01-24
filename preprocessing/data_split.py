import random
import shutil
from pathlib import Path


#configuration
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15

#class folder names inside the source dataset directory
CLASSES = ["rock", "paper", "scissors"]

#project paths (relative to this script file)
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "dataset"
DST_DIR = BASE_DIR / "dataset_splits"   

#allowed image file extensions
IMG_EXTS = {".jpg", ".jpeg", ".png"}

#returning a sorted list of image file paths with allowed extensions
def list_images(folder: Path):
    return (
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    )

#main function
def main():
    random.seed(SEED)
    
    #creating the destination directory structure (train/val/test for each class)
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (DST_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    #splitting and copying files for each class
    for cls in CLASSES:
        
        #listing all images for the current class
        files = list_images(SRC_DIR / cls)
        
        #shuffling files before splitting
        random.shuffle(files)
        
        #computing split sizes
        n = len(files)
        n_test = int(n * TEST_FRAC)
        n_val = int(n * VAL_FRAC)
        
        
        #dividing the shuffled list into test/val/train subsets
        splits = {
            "test": files[:n_test],
            "val":  files[n_test:n_test + n_val],
            "train": files[n_test + n_val:]
        }
        
        #copying each subset into the corresponding destination folder
        for split_name, file_list in splits.items():
            for src_path in file_list:
                dst_path = DST_DIR / split_name / cls / src_path.name
                shutil.copy2(src_path, dst_path)
    
    #printing final message
    print("Split complete")
    print(f"Output folder: {DST_DIR}")
    
#(running the script only when executed directly)
if __name__ == "__main__":
    main()
