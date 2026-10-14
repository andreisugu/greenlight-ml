"""Helper script to download TMDB 5000 movie dataset from Kaggle."""
import os
import shutil
import kagglehub

def download_data():
    print("Downloading TMDB 5000 dataset (tmdb/tmdb-movie-metadata)...")
    path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    print(f"Downloaded to cache: {path}")

    target_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            src = os.path.join(path, filename)
            dst = os.path.join(target_dir, filename)
            shutil.copy2(src, dst)
            print(f"Extracted: {filename} -> {dst}")

if __name__ == "__main__":
    download_data()
