from huggingface_hub import snapshot_download
from pathlib import Path

_REPO_ID = "anirudh1112/corrected-tobacco-dataset-with-ocr"
_REPO_TYPE="dataset"
_REVISION = "c45813f9ac080e49c08d4876145e897139364cd5"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_DIR = _PROJECT_ROOT/"data/raw"
_IGNORE_PATTERNS = "*.hocr"

def download_raw_dataset(repo_id : str = _REPO_ID, 
                         repo_type : str =  _REPO_TYPE, 
                         revision: str = _REVISION, 
                         local_dir : Path | str = _LOCAL_DIR, 
                         ignore_patterns : str = _IGNORE_PATTERNS) -> str:
    
    return snapshot_download(repo_id = repo_id,
                             repo_type = repo_type,
                             revision = revision,
                             local_dir = local_dir,
                             ignore_patterns = ignore_patterns)

if __name__ == "__main__":
    print(download_raw_dataset())