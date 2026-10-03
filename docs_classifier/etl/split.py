from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


_ROOT_FOLDER = Path(__file__).resolve().parents[2]/"data/raw"
_META_DATA_FILE = "metadata.jsonl"
_FOLDERS = ("train", "test", "val")
_OUTPUT_FOLDER = Path(__file__).resolve().parents[2]/"data_splits"
_COLUMNS = ("label", "path", "split")
_SEED = 42
_TEST_SPLIT_RATE1 = 0.3 # To split the dataset into a 70%-30% split so that 70% goes into the training set. 
_TEST_SPLIT_RATE2 = 0.5 # To split the 30% into 15% and 15% so that 15% goes to “test” and 15% to “val”
_N_EXPECTED = 2737
_N_CLASSES = 10
_N_SPLITS = 3

def load_metadata(root_folder: Path = _ROOT_FOLDER, folders : tuple[str, ...] = _FOLDERS, meta_data_file : str = _META_DATA_FILE) -> pd.DataFrame:
    dfs = []
    for f in folders:
        d = pd.read_json(root_folder/f"{f}/{meta_data_file}", lines = True)
        d["path"] = f+"/"+d["file_name"]
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)    
    return df


def split_dataset(df: pd.DataFrame, test_split_rate1: float = _TEST_SPLIT_RATE1, test_split_rate2: float = _TEST_SPLIT_RATE2, seed: int = _SEED) -> pd.DataFrame:
    
    train, temp = train_test_split(df, test_size = test_split_rate1, random_state = seed, stratify = df["label"])
    val, test = train_test_split(temp, test_size = test_split_rate2, random_state = seed, stratify = temp["label"])
    
    train = train.assign(split = "train")
    val = val.assign(split = "val")
    test = test.assign(split = "test")
    
    df = pd.concat([train, val, test], ignore_index=True)
    
    return df


def verify_dataset(df: pd.DataFrame, n_expected: int = _N_EXPECTED, n_classes: int = _N_CLASSES, n_splits: int = _N_SPLITS) -> None:
    duplicates = df["file_name"].duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate filenames in the dataset."
    
    assert df.shape[0] == n_expected, f"Expected {n_expected} rows in the dataset, but found {df.shape[0]}."
    
    assert df["split"].nunique() == n_splits, f"Expected {n_splits} splits in the dataset, but found {df['split'].nunique()}."
    
    for name, group in df.groupby("split"):
        n = group["label"].nunique()
        assert n == n_classes, f"Expected {n_classes} classes in the '{name}' split, but found {n}."

def save_dataset(df: pd.DataFrame, output_folder: Path = _OUTPUT_FOLDER, columns: tuple[str, ...] = _COLUMNS) -> None:
    df = df[list(columns)]
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder/"data_splits.csv"
    df.to_csv(output_path, index=False)


    

