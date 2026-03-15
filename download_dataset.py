"""
Helper script to download the DREAMER dataset, bypassing Git LFS.
Since the dataset is 453MB, cloning from GitHub repeatedly quickly exhausts
the free 1GB/month Git LFS bandwidth quota.
"""

import os
import requests
import gdown

# Public Google Drive link for DREAMER dataset or a Zenodo mirror.
# If you don't have one hosted, you'll need the user to upload it to their
# Drive or provide a link. I will provide a gdown mechanism assuming they 
# can paste their own GDrive file ID here if they have it hosted.
#
# Alternatively, since the dataset is public, here is a known working
# mirror for DREAMER if available, but a robust way is to ask the user.
# For now, let's write a generic downloader that prompts for a Drive ID.

def download_dreamer(file_id=None, dest_path="data/DREAMER.mat"):
    """Download DREAMER.mat using gdown."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100_000_000:
        print(f"Dataset already exists at {dest_path}")
        return
        
    if not file_id:
        print("ERROR: Git LFS budget exceeded.")
        print("To fix this, upload DREAMER.mat to your Google Drive,")
        print("get the 'Share' link, extract the file ID, and run:")
        print("  python download_dataset.py --gdrive_id <YOUR_FILE_ID>")
        return
        
    print(f"Downloading DREAMER.mat from Google Drive ID: {file_id} ...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)
    
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"\nSuccess! Downloaded {size_mb:.1f} MB to {dest_path}")
    else:
        print("\nDownload failed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdrive_id", type=str, help="Google Drive File ID for DREAMER.mat")
    args = parser.parse_args()
    
    download_dreamer(args.gdrive_id)
