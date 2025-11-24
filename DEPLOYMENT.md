# 🚀 Deployment Guide: Hugging Face Spaces

## Step-by-Step Instructions

### Option 1: Upload via Web Interface (Easiest)

1. **Go to your Space:**
   - Visit: https://huggingface.co/spaces/AhiBucket/Hand-wave
   - Click on "Files" tab

2. **Upload the following files:**
   - `app.py` (your main application)
   - `functions.py` (physics engine)
   - `requirements.txt` (dependencies)
   - `README.md` (project documentation)

3. **Important Files to Upload:**
   ```
   ✅ app.py
   ✅ functions.py
   ✅ requirements.txt
   ✅ README.md
   ```

4. **Files NOT needed for deployment** (optional, for documentation only):
   ```
   ❌ verify_physics.py
   ❌ verify_comparison.py
   ❌ Comparison_Notebook.ipynb
   ❌ ModularedPresentation.ipynb
   ```

5. **Wait for Build:**
   - Hugging Face will automatically rebuild your Space
   - Check the "Logs" tab to monitor progress
   - Build typically takes 2-5 minutes

### Option 2: Git Push (Advanced)

If you have Git installed:

```bash
# Navigate to your project directory
cd "d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\psi_solve2"

# Initialize git (if not already done)
git init

# Add Hugging Face remote
git remote add hf https://huggingface.co/spaces/AhiBucket/Hand-wave

# Add files
git add app.py functions.py requirements.txt README.md .gitignore

# Commit
git commit -m "Deploy enhanced Quantum Solver with professional UI"

# Push to Hugging Face
git push hf main
```

## 📋 Pre-Deployment Checklist

- [x] `app.py` - Enhanced with Plotly and professional UI
- [x] `functions.py` - Physics engine (already exists)
- [x] `requirements.txt` - All dependencies listed
- [x] `README.md` - Professional documentation
- [ ] Test locally first: `streamlit run app.py`

## 🧪 Test Locally Before Deploying

```bash
# In your project directory
streamlit run app.py
```

This will open the app at `http://localhost:8501`

## 🔧 Troubleshooting

### If build fails:

1. **Check requirements.txt:**
   - Make sure all packages are spelled correctly
   - Use compatible versions

2. **Check logs:**
   - Go to your Space → "Logs" tab
   - Look for error messages

3. **Common issues:**
   - **MediaPipe error:** If camera features fail, it's OK - they won't work on Hugging Face anyway (no camera access)
   - **Memory issues:** Reduce `N_GRID` in app.py if needed

### If you see "opencv-python" error:

Replace in `requirements.txt`:
```
opencv-python-headless
```

## 📝 Post-Deployment

1. **Test your Space:**
   - Visit: https://huggingface.co/spaces/AhiBucket/Hand-wave
   - Try all three pages (Simulator, Benchmarks, Theory)
   - Test different potentials

2. **Share your work:**
   - The Space URL is public
   - Add it to your CV/resume
   - Share with professors and admissions committees

## 🎓 For Academic Applications

When sharing with Oxford/Cambridge:

1. **Highlight the Verification:**
   - Show the "Benchmarks & Verification" page
   - Emphasize <0.003% error on analytical cases
   - Mention QMSolve cross-validation

2. **Emphasize the Theory:**
   - The "Theory & Method" page shows mathematical rigor
   - Demonstrates understanding of numerical methods

3. **Professional Presentation:**
   - Clean, interactive UI
   - Well-documented code
   - Comprehensive testing

## 🔗 Quick Links

- Your Space: https://huggingface.co/spaces/AhiBucket/Hand-wave
- Build Logs: https://huggingface.co/spaces/AhiBucket/Hand-wave?logs=build
- Hugging Face Docs: https://huggingface.co/docs/hub/spaces

---

**Ready to deploy!** Just upload the 4 main files via the web interface.
