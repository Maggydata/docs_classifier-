import pytesseract
from PIL import Image
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

_TESSERACT_LANG = "eng"
_TESSERACT_CONFIG = "--oem 1 --psm 3"

_ROOT_PATH = Path(__file__).resolve().parents[3]
_OCR_CACHE_DIR = "data/interim/ocr"
_SPLIT_CSV = "data_splits/data_splits.csv"
_IMAGES_DIR = "data/raw"
_OCR_PARQUET = "data/interim/ocr.parquet"

def run_tesseract(img : Image.Image, tesseract_lang: str = _TESSERACT_LANG, tesseract_config: str = _TESSERACT_CONFIG) -> dict[str, list]:
    
    """Run Tesseract on an image and return the raw output from `image_to_data`.

    A single OCR call: the text and confidence scores will be calculated from this result.
    """
    
    data = pytesseract.image_to_data(
        img,
        lang = tesseract_lang,
        config = tesseract_config,
        output_type = pytesseract.Output.DICT
    )
    
    return data

def parse_tesseract_output(raw: dict[str, list]) -> dict:
    """Calculates the text, average confidence, and number of words
    based on the raw output of image_to_data (Output.DICT).

    Elements that are not words (page, block, paragraph, line)
    have a confidence score of -1 and are excluded, as are empty texts.
    """
    
    lines = {}
    confidences = []
    
    for i, word in enumerate(raw["text"]):
        conf = float(raw["conf"][i])
        word = word.strip()
        if conf < 0 or not word:
            continue
        
        key = (raw["block_num"][i], raw["par_num"][i], raw["line_num"][i])
        lines.setdefault(key, []).append(word)
        confidences.append(conf)
        
    text = "\n".join(" ".join(words) for words in lines.values())
        
    text_data = {
            "text": text,
            "mean_confidence" : sum(confidences) / len(confidences) if confidences else None,
            "nb_words" : len(confidences)
        }
    
    return text_data



def ocr_image(img : Image.Image, tesseract_version: str,  tesseract_lang: str = _TESSERACT_LANG, tesseract_config: str = _TESSERACT_CONFIG) -> dict:
    """Full OCR of an image: a single call to Tesseract, followed by calculation of the result."""
    
    raw = run_tesseract(img, tesseract_lang, tesseract_config)
    text_data = parse_tesseract_output(raw)
    text_data.update({
        "tesseract_version" : tesseract_version,
        "tesseract_lang" : tesseract_lang,
        "tesseract_config" : tesseract_config
    })
    
    return text_data

def cache_path(doc_path: str | Path, cache_dir: Path = _ROOT_PATH/_OCR_CACHE_DIR) -> Path:
    """The JSON path of a document.

    Takes the relative path of the image and changes the file extension:
    train/87150355.jpg -> data/interim/ocr/train/87150355.json
    """
    return cache_dir / Path(doc_path).with_suffix(".json")

def write_cache(doc_path: str | Path, text_data : dict, cache_dir: Path = _ROOT_PATH/_OCR_CACHE_DIR) -> None:
    """Write the cache file for a given image path."""
    
    path = cache_path(doc_path, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding = "utf-8") as f:
        json.dump(text_data, f, ensure_ascii = False, indent = 2)
    tmp.replace(path)
    

def run_ocr(split_csv : Path = _ROOT_PATH/_SPLIT_CSV, limit: int | None = None, 
            cache_dir : Path = _ROOT_PATH/_OCR_CACHE_DIR, images_dir : Path = _ROOT_PATH/_IMAGES_DIR,
            tesseract_lang: str = _TESSERACT_LANG, tesseract_config: str = _TESSERACT_CONFIG, )-> dict :
    
    """Runs OCR on all documents in the split CSV file.

    Documents already in the cache are skipped. An error on a document
    is reported without stopping the processing of the others.
    `limit` allows you to process only the first n documents (for testing).
    """
    tesseract_version = str(pytesseract.get_tesseract_version())
    df = pd.read_csv(split_csv)
    if limit is not None:
        df = df.head(limit)
        
    n_done, n_skipped = 0,0
    errors = []
    
    for _, line in tqdm(df.iterrows(), total = len(df), desc = "OCR"):
        
        doc_path = line["path"]
        doc_output_path = cache_path(doc_path, cache_dir)
        
        if doc_output_path.exists() : 
            n_skipped += 1
            continue
        
        try :
            with Image.open(images_dir/doc_path) as img :
                text_data = ocr_image(img, tesseract_version, tesseract_lang, tesseract_config) 
            
            text_data.update({
                "path" : doc_path
            })
            
            write_cache(doc_path, text_data, cache_dir)
            n_done += 1
            
        except Exception as e :
            errors.append({"path" : doc_path, "error" : f"{type(e).__name__} : {e}"})
    
    print(f"processed : {n_done} | already in cache : {n_skipped} | Errors : {len(errors)}")
    for err in errors:
        print(f"  ✗ {err['path']} — {err['error']}")            
                 
    return {"done": n_done, "skipped": n_skipped, "errors": errors}  
    

def consolidate_ocr(split_csv: Path = _ROOT_PATH/_SPLIT_CSV, cache_dir : Path = _ROOT_PATH/_OCR_CACHE_DIR, output: Path = _ROOT_PATH/_OCR_PARQUET) -> pd.DataFrame:
    """Combine all the JSON data from the OCR cache into a single Parquet file,
    appended to the split CSV to retrieve the label and split."""

    json_files = sorted(cache_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON found in {cache_dir}")

    records = []
    for f in json_files:
        with open(f, encoding="utf-8") as fh:
            records.append(json.load(fh))
    ocr_df = pd.DataFrame(records)

    splits = pd.read_csv(split_csv)[["path", "label", "split"]]
    df = splits.merge(ocr_df, on="path", how="left", validate="one_to_one")

    missing = df["text"].isna().sum()
    print(f"JSON read : {len(ocr_df)} | CSV's Doc : {len(splits)} | without OCR : {missing}")

    
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Parquet written : {output}")

    return df

if __name__ == "__main__":
    run_ocr()
    consolidate_ocr()