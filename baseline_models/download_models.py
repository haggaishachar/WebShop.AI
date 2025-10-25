"""
Download pre-trained baseline models from Google Drive.

This script downloads the IL (Imitation Learning) model checkpoints:
- choice_il_epoch9.pth (BERT-based choice model)
- search_il_checkpoints_800.zip (BART-based search model)

Model URLs are from the official WebShop repository:
https://drive.google.com/drive/folders/1liZmB1J38yY_zsokJAxRfN8xVO1B_YmD

Directory structure after download:
    baseline_models/
        ckpts/
            web_click/
                epoch_9/
                    model.pth
            web_search/
                checkpoint-800/
                    (BART model files)
"""

import os
import sys
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Error: gdown is not installed.")
    print("Please install it with: uv pip install gdown")
    sys.exit(1)


# Google Drive file IDs from the WebShop repository
CHOICE_MODEL_ID = "1liZmB1J38yY_zsokJAxRfN8xVO1B_YmD"  # Folder ID
CHOICE_MODEL_FILE = "choice_il_epoch9.pth"
SEARCH_MODEL_FILE = "search_il_checkpoints_800.zip"

# Local paths
SCRIPT_DIR = Path(__file__).parent
CKPTS_DIR = SCRIPT_DIR / "ckpts"
CHOICE_MODEL_DIR = CKPTS_DIR / "web_click" / "epoch_9"
SEARCH_MODEL_DIR = CKPTS_DIR / "web_search"


def download_models():
    """Download and extract the baseline IL models."""
    
    print("=" * 80)
    print("WebShop Baseline Models Downloader")
    print("=" * 80)
    print()
    
    # Create directories
    CKPTS_DIR.mkdir(exist_ok=True)
    CHOICE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if models already exist
    choice_model_path = CHOICE_MODEL_DIR / "model.pth"
    search_model_path = SEARCH_MODEL_DIR / "checkpoint-800"
    
    if choice_model_path.exists() and search_model_path.exists():
        print("✓ Models already downloaded!")
        print(f"  - Choice model: {choice_model_path}")
        print(f"  - Search model: {search_model_path}")
        print()
        print("To re-download, delete the ckpts/ directory and run this script again.")
        return
    
    print("Downloading baseline IL models from Google Drive...")
    print("This may take a few minutes depending on your connection.")
    print()
    
    # Download the entire folder from Google Drive
    # The folder contains both choice_il_epoch9.pth and search_il_checkpoints_800.zip
    temp_dir = CKPTS_DIR / "temp_download"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        print("📥 Downloading models folder from Google Drive...")
        gdown.download_folder(
            id=CHOICE_MODEL_ID,
            output=str(temp_dir),
            quiet=False,
            use_cookies=False
        )
        
        # Find and move the downloaded files
        downloaded_files = list(temp_dir.rglob("*"))
        choice_pth = None
        search_zip = None
        
        for file in downloaded_files:
            if file.is_file():
                if "choice" in file.name.lower() and file.suffix == ".pth":
                    choice_pth = file
                elif "search" in file.name.lower() and file.suffix == ".zip":
                    search_zip = file
        
        # Process choice model
        if choice_pth and not choice_model_path.exists():
            print(f"\n📦 Processing choice model: {choice_pth.name}")
            choice_pth.rename(choice_model_path)
            print(f"✓ Choice model saved to: {choice_model_path}")
        elif not choice_model_path.exists():
            print("⚠️  Warning: choice_il_epoch9.pth not found in download.")
        
        # Process search model
        if search_zip and not search_model_path.exists():
            print(f"\n📦 Extracting search model: {search_zip.name}")
            with zipfile.ZipFile(search_zip, 'r') as zip_ref:
                zip_ref.extractall(SEARCH_MODEL_DIR)
            
            # The zip might contain a checkpoint-800 folder, or files directly
            # Try to find the checkpoint-800 folder
            checkpoint_dirs = list(SEARCH_MODEL_DIR.glob("*checkpoint*"))
            if checkpoint_dirs:
                if checkpoint_dirs[0].name != "checkpoint-800":
                    checkpoint_dirs[0].rename(search_model_path)
                print(f"✓ Search model extracted to: {search_model_path}")
            else:
                # Files might be extracted directly, create the folder structure
                extracted_files = [f for f in SEARCH_MODEL_DIR.iterdir() 
                                 if f.is_file() and f.suffix in ['.json', '.bin', '.txt']]
                if extracted_files:
                    search_model_path.mkdir(exist_ok=True)
                    for f in extracted_files:
                        f.rename(search_model_path / f.name)
                    print(f"✓ Search model extracted to: {search_model_path}")
        elif not search_model_path.exists():
            print("⚠️  Warning: search_il_checkpoints_800.zip not found in download.")
        
    except Exception as e:
        print(f"\n❌ Error downloading models: {e}")
        print("\nYou can manually download the models from:")
        print("https://drive.google.com/drive/folders/1liZmB1J38yY_zsokJAxRfN8xVO1B_YmD")
        print()
        print("Then organize them as:")
        print(f"  {choice_model_path}")
        print(f"  {search_model_path}/")
        sys.exit(1)
    finally:
        # Clean up temp directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n" + "=" * 80)
    print("✓ Download complete!")
    print("=" * 80)
    print()
    print("Models are ready to use. You can now run:")
    print("  make run-web-agent-il")
    print()


def verify_models():
    """Verify that models are downloaded and accessible."""
    choice_model_path = CHOICE_MODEL_DIR / "model.pth"
    search_model_path = SEARCH_MODEL_DIR / "checkpoint-800"
    
    if not choice_model_path.exists():
        print(f"❌ Choice model not found: {choice_model_path}")
        return False
    
    if not search_model_path.exists():
        print(f"❌ Search model not found: {search_model_path}")
        return False
    
    # Check if search model has required files
    required_files = ["config.json", "pytorch_model.bin"]
    missing_files = []
    for file in required_files:
        if not (search_model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Search model missing files: {', '.join(missing_files)}")
        return False
    
    print("✓ All models are present and ready!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download WebShop baseline IL models")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify models are downloaded (don't download)"
    )
    args = parser.parse_args()
    
    if args.verify:
        verify_models()
    else:
        download_models()


