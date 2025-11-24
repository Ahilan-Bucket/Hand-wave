# Quick Deployment to Hugging Face Space
# This script helps you sync your code to Hugging Face

## OPTION 1: Upload via Python Script (Easiest)

1. Get your Hugging Face token:
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token" → Create a WRITE token
   - Copy the token

2. Run the upload script:
   ```bash
   cd d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\Hand-wave
   python upload_to_hf.py
   ```
   
3. Paste your token when prompted

4. Done! Check your Space: https://huggingface.co/spaces/AhiBucket/Hand-wave

---

## OPTION 2: Clone HF Space and Push (Alternative)

If Option 1 doesn't work, try this:

```bash
# 1. Clone the Hugging Face Space
cd d:\Documents\SFU\PHYS385_Quantum2\CodeProjects
git clone https://huggingface.co/spaces/AhiBucket/Hand-wave hf-hand-wave

# 2. Copy files
cd hf-hand-wave
Copy-Item -Path "..\Hand-wave\app.py" -Destination "." -Force
Copy-Item -Path "..\Hand-wave\functions.py" -Destination "." -Force  
Copy-Item -Path "..\Hand-wave\requirements.txt" -Destination "." -Force
Copy-Item -Path "..\Hand-wave\README.md" -Destination "." -Force

# 3. Commit and push
git add .
git commit -m "Fix: Plotly compatibility for wavefunction display"
git push
```

When prompted for credentials:
- Username: Your Hugging Face username
- Password: Your Hugging Face WRITE token (NOT your password!)

---

## Files Ready for Deployment ✅

All files have been prepared in: `d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\Hand-wave`

- ✅ app.py (with Plotly fixes)
- ✅ functions.py (physics engine)
- ✅ requirements.txt (dependencies)
- ✅ README.md (documentation)
- ✅ .gitignore

## What Was Fixed 🔧

The main issue causing wavefunctions not to display on Hugging Face:

1. **Array dimension validation** - Ensures x and y arrays match in length
2. **Data type conversion** - Converts numpy arrays to lists for Plotly compatibility
3. **Explicit array handling** - Uses x_internal explicitly instead of inline slicing

These fixes ensure the app works on both local Streamlit and Hugging Face Spaces!
