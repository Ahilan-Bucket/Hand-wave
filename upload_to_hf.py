"""
Script to upload files to Hugging Face Space
Run this after getting your HF token from: https://huggingface.co/settings/tokens
"""

from huggingface_hub import HfApi, login
import os

def upload_to_hf_space():
    print("=" * 60)
    print("Hugging Face Space Uploader")
    print("=" * 60)
    
    # Get token from user
    print("\n1. Get your token from: https://huggingface.co/settings/tokens")
    print("   (Create a WRITE token if you don't have one)")
    token = input("\n2. Paste your Hugging Face token here: ").strip()
    
    if not token:
        print("❌ No token provided. Exiting.")
        return
    
    try:
        # Login
        print("\n3. Logging in to Hugging Face...")
        login(token=token)
        print("✅ Login successful!")
        
        # Initialize API
        api = HfApi()
        
        # Space details
        repo_id = "AhiBucket/Hand-wave"
        repo_type = "space"
        
        print(f"\n4. Uploading files to {repo_id}...")
        
        # Files to upload
        files_to_upload = [
            "app.py",
            "functions.py",
            "requirements.txt",
            "README.md",
            ".gitignore"
        ]
        
        # Upload each file
        for file in files_to_upload:
            if os.path.exists(file):
                print(f"   📤 Uploading {file}...")
                api.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=file,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
                print(f"   ✅ {file} uploaded!")
            else:
                print(f"   ⚠️  {file} not found, skipping...")
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! All files uploaded to Hugging Face Space!")
        print("=" * 60)
        print(f"\n🌐 View your Space at: https://huggingface.co/spaces/{repo_id}")
        print("\n⏳ The Space will rebuild automatically (may take 1-2 minutes)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Make sure your token has WRITE permissions")
        print("- Verify the Space exists: https://huggingface.co/spaces/AhiBucket/Hand-wave")
        print("- Check your internet connection")

if __name__ == "__main__":
    upload_to_hf_space()
