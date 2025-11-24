# Deployment Guide: Syncing GitHub to Hugging Face Space

## Quick Setup (Choose ONE method)

### Method 1: Direct Push to Hugging Face (Recommended)

1. **Get your Hugging Face Token:**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token" → Create a **Write** token
   - Copy the token

2. **Login to Hugging Face CLI:**
   ```bash
   huggingface-cli login
   ```
   - Paste your token when prompted

3. **Clone the Hugging Face Space repository:**
   ```bash
   cd d:\Documents\SFU\PHYS385_Quantum2\CodeProjects
   git clone https://huggingface.co/spaces/AhiBucket/Hand-wave huggingface-hand-wave
   ```

4. **Copy files from GitHub repo to HF Space:**
   ```bash
   cd huggingface-hand-wave
   Copy-Item -Path "..\Hand-wave\app.py" -Destination "." -Force
   Copy-Item -Path "..\Hand-wave\functions.py" -Destination "." -Force
   Copy-Item -Path "..\Hand-wave\requirements.txt" -Destination "." -Force
   Copy-Item -Path "..\Hand-wave\README.md" -Destination "." -Force
   ```

5. **Commit and push to Hugging Face:**
   ```bash
   git add .
   git commit -m "Update: Fixed Plotly compatibility for wavefunction display"
   git push
   ```

---

### Method 2: GitHub Integration (Via Web UI)

1. **Go to your Space Settings:**
   - Navigate to https://huggingface.co/spaces/AhiBucket/Hand-wave
   - Click **"Settings"** (top right, near your profile)

2. **Connect GitHub Repository:**
   - Scroll to **"Repository"** section
   - Click **"Connect to GitHub"**
   - Select repository: `Ahilan-Bucket/Hand-wave`
   - Choose branch: `main`
   - Click **"Save"**

3. **Enable Auto-sync:**
   - Toggle **"Automatically rebuild on push"**
   - This will sync your GitHub repo to HF Space automatically

---

## What's Been Done Already ✅

1. ✅ Created new GitHub repository: `Hand-wave`
2. ✅ Copied all necessary files from `psi_solve2`
3. ✅ Fixed Plotly compatibility issues for Hugging Face
4. ✅ Pushed code to GitHub: https://github.com/Ahilan-Bucket/Hand-wave

## What's Next 🚀

- Choose Method 1 or Method 2 above to sync to Hugging Face
- Once synced, the wavefunctions should display correctly!
- Test the deployment at: https://huggingface.co/spaces/AhiBucket/Hand-wave

## Troubleshooting

**If the app doesn't start:**
- Check the build logs in the Hugging Face Space
- Verify `requirements.txt` has all dependencies
- Make sure `app.py` is in the root directory

**If wavefunctions still don't show:**
- The fixes we made should resolve this
- Check browser console for JavaScript errors
- Verify Plotly version compatibility
