import os

# ----- Dataset roots ---------------------------------------------------------

CHEXPERT_ROOT = os.environ.get(
    "CHEXPERT_ROOT",
    "/path/to/CheXpert-v1.0",
)

CHESTXRAY14_ROOT = os.environ.get(
    "CHESTXRAY14_ROOT",
    "/path/to/ChestXRay14",
)

ODIR_ROOT = os.environ.get(
    "ODIR_ROOT",
    "/path/to/ODIR-5K",
)

# ----- LabCora outputs -------------------------------------------------------

OUTPUT_DIR = os.environ.get(
    "LABCORA_OUTPUT",
    "./outputs",
)

# ----- CheXpert sub-paths ----------------------------------------------------

CHEXPERT_SPLIT_DIR = os.path.join(CHEXPERT_ROOT, "splits")
CHEXPERT_TRAIN_CSV = os.path.join(CHEXPERT_SPLIT_DIR, "train.csv")
CHEXPERT_VALID_CSV = os.path.join(CHEXPERT_SPLIT_DIR, "valid.csv")
CHEXPERT_TEST_CSV = os.path.join(CHEXPERT_SPLIT_DIR, "test.csv")

# ----- ChestXRay14 sub-paths -------------------------------------------------

CHESTXRAY14_ENTRY_CSV = os.path.join(
    CHESTXRAY14_ROOT,
    "Data_Entry_2017_v2020.csv",
)
CHESTXRAY14_IMAGE_DIR = os.path.join(
    CHESTXRAY14_ROOT,
    "Total_Images",
    "images",
)
CHESTXRAY14_PREFIX = "ChestXRay14"

# ----- ODIR-5K sub-paths -----------------------------------------------------

ODIR_ALL_CSV = os.path.join(ODIR_ROOT, "All.csv")
ODIR_TRAIN_DIR = os.path.join(ODIR_ROOT, "train")
ODIR_VALID_DIR = os.path.join(ODIR_ROOT, "valid")
ODIR_TEST_DIR = os.path.join(ODIR_ROOT, "test")
ODIR_SPLIT_DIR = os.path.join(ODIR_ROOT, "splits")
