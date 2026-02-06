import os
import requests
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://olmo-data.org"
OUTPUT_DIR = Path("/data/user_data/nsrikant/bbox_data/olmo3_eval")
URL_FILE = "/home/nsrikant/BehaviorBoxNew/v3-small-ppl-validation.txt"


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name,
                leave=False
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error: {e}")
        return False


def load_urls_from_file(filepath: str) -> list:
    """Load URL paths from a text file."""
    paths = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().split(",")[-1]
            line = line.replace("v3_small_{TOKENIZER}", "v3_small_dolma2-tokenizer")
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                paths.append(line)
    return paths


def main():
    # Load paths from file
    if not Path(URL_FILE).exists():
        print(f"Error: {URL_FILE} not found!")
        return
    
    datasets = load_urls_from_file(URL_FILE)
    print(f"Downloading {len(datasets)} files to {OUTPUT_DIR}\n")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for i, dataset_path in enumerate(datasets, 1):
        url = f"{BASE_URL}/{dataset_path}"
        output_path = OUTPUT_DIR / dataset_path
        
        print(f"[{i}/{len(datasets)}] {dataset_path}")
        
        if output_path.exists():
            print(f"  → Skipped (already exists)")
            skipped += 1
            continue
        
        if download_file(url, output_path):
            print(f"  ✓ Downloaded successfully")
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Download complete!")
    print(f"  Successful: {successful}")
    print(f"  Skipped:    {skipped}")
    print(f"  Failed:     {failed}")


if __name__ == "__main__":
    main()