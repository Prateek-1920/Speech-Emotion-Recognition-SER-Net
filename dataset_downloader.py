# dataset_downloader.py

import os
import requests
import zipfile
import shutil
import argparse
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(destination, 'wb') as file, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def setup_ravdess():
    """Download and setup the RAVDESS dataset"""
    print("\n--- Setting up RAVDESS dataset ---")
    
    # Create directory if it doesn't exist
    if not os.path.exists("RAVDESS"):
        os.makedirs("RAVDESS")
    
    # Download URL
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = "RAVDESS/Audio_Speech_Actors_01-24.zip"
    
    # Download if file doesn't exist
    if not os.path.exists(zip_path):
        print(f"Downloading RAVDESS dataset from {url}")
        success = download_file(url, zip_path)
        if not success:
            print("Failed to download RAVDESS dataset.")
            return False
    else:
        print(f"Found existing RAVDESS download at {zip_path}")
    
    # Extract zip file
    try:
        print("Extracting RAVDESS dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("RAVDESS")
        
        print("RAVDESS dataset setup complete!")
        return True
    except Exception as e:
        print(f"Error extracting RAVDESS dataset: {e}")
        return False

def setup_tess():
    """Download and setup the TESS dataset"""
    print("\n--- Setting up TESS dataset ---")
    
    # Create directory if it doesn't exist
    if not os.path.exists("TESS"):
        os.makedirs("TESS")
    
    # Download URL
    url = "https://tspace.library.utoronto.ca/bitstream/1807/24487/1/tess toronto emotional speech set data.zip"
    zip_path = "TESS/tess.zip"
    
    # Download if file doesn't exist
    if not os.path.exists(zip_path):
        print(f"Downloading TESS dataset from {url}")
        success = download_file(url, zip_path)
        if not success:
            print("Failed to download TESS dataset.")
            return False
    else:
        print(f"Found existing TESS download at {zip_path}")
    
    # Extract zip file
    try:
        print("Extracting TESS dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("TESS")
        
        print("TESS dataset setup complete!")
        return True
    except Exception as e:
        print(f"Error extracting TESS dataset: {e}")
        return False

def setup_cremad():
    """Download and setup the CREMA-D dataset"""
    print("\n--- Setting up CREMA-D dataset ---")
    
    # Create directory if it doesn't exist
    if not os.path.exists("CREMA-D"):
        os.makedirs("CREMA-D")
    
    # Download URL
    url = "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip"
    zip_path = "CREMA-D/cremad.zip"
    
    # Download if file doesn't exist
    if not os.path.exists(zip_path):
        print(f"Downloading CREMA-D dataset from {url}")
        success = download_file(url, zip_path)
        if not success:
            print("Failed to download CREMA-D dataset.")
            return False
    else:
        print(f"Found existing CREMA-D download at {zip_path}")
    
    # Extract zip file
    try:
        print("Extracting CREMA-D dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("CREMA-D")
        
        # Move the AudioWAV folder to the main CREMA-D directory
        src_dir = os.path.join("CREMA-D", "CREMA-D-master", "AudioWAV")
        dest_dir = os.path.join("CREMA-D", "AudioWAV")
        
        if os.path.exists(src_dir) and not os.path.exists(dest_dir):
            print("Moving AudioWAV folder to main CREMA-D directory...")
            shutil.move(src_dir, "CREMA-D")
        
        print("CREMA-D dataset setup complete!")
        return True
    except Exception as e:
        print(f"Error extracting CREMA-D dataset: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and setup emotion datasets for SER")
    parser.add_argument("--ravdess", action="store_true", help="Download RAVDESS dataset")
    parser.add_argument("--tess", action="store_true", help="Download TESS dataset")
    parser.add_argument("--cremad", action="store_true", help="Download CREMA-D dataset")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    
    args = parser.parse_args()
    
    # If no arguments are provided, download all datasets
    if not any(vars(args).values()):
        args.all = True
    
    # Track success of downloads
    success_count = 0
    total_attempts = 0
    
    if args.all or args.ravdess:
        total_attempts += 1
        if setup_ravdess():
            success_count += 1
    
    if args.all or args.tess:
        total_attempts += 1
        if setup_tess():
            success_count += 1
    
    if args.all or args.cremad:
        total_attempts += 1
        if setup_cremad():
            success_count += 1
    
    print(f"\nDataset setup complete! {success_count}/{total_attempts} datasets were set up successfully.")
    print("You can now use them with the SER system.")

if __name__ == "__main__":
    main()
