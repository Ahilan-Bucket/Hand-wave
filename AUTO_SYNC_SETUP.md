# 🔄 Automatic GitHub → Hugging Face Sync Setup

This guide will set up **automatic synchronization** so that every commit to GitHub automatically updates your Hugging Face Space.

## ✨ What This Does

- **Commit to GitHub** → Automatically pushes to Hugging Face
- **Zero manual work** after initial setup
- **Instant deployment** - changes go live automatically
- **Works with any branch** (configured for `main`)

---

## 🚀 One-Time Setup (5 minutes)

### Step 1: Get Your Hugging Face Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it: `GitHub Actions Sync`
4. Permission: Select **"Write"**
5. Click **"Generate token"**
6. **COPY THE TOKEN** (you won't see it again!)

### Step 2: Add Token to GitHub Secrets

1. Go to your GitHub repository: https://github.com/Ahilan-Bucket/Hand-wave
2. Click **"Settings"** (top menu)
3. In the left sidebar, click **"Secrets and variables"** → **"Actions"**
4. Click **"New repository secret"**
5. Name: `HF_TOKEN`
6. Value: Paste your Hugging Face token
7. Click **"Add secret"**

### Step 3: Push the GitHub Actions Workflow

```bash
cd d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\Hand-wave

# Add the workflow file
git add .github/workflows/sync-to-huggingface.yml

# Commit
git commit -m "Add automatic sync to Hugging Face Space"

# Push to GitHub
git push origin main
```

### Step 4: Verify It Works

1. Go to: https://github.com/Ahilan-Bucket/Hand-wave/actions
2. You should see the workflow running
3. Wait for it to complete (green checkmark ✅)
4. Check your Hugging Face Space: https://huggingface.co/spaces/AhiBucket/Hand-wave

---

## 🎯 How It Works

```
┌─────────────────┐
│  You commit to  │
│     GitHub      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GitHub Actions  │
│  automatically  │
│    triggers     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Pushes to     │
│ Hugging Face    │
│     Space       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Space rebuilds │
│   automatically │
│   (1-2 minutes) │
└─────────────────┘
```

---

## 📝 Daily Workflow (After Setup)

From now on, your workflow is simple:

```bash
# 1. Make changes to your code
# (edit app.py, functions.py, etc.)

# 2. Commit to GitHub
git add .
git commit -m "Your change description"
git push origin main

# 3. That's it! 🎉
# GitHub Actions automatically syncs to Hugging Face
# Check progress at: https://github.com/Ahilan-Bucket/Hand-wave/actions
```

---

## 🔍 Monitoring

### Check GitHub Actions Status:
https://github.com/Ahilan-Bucket/Hand-wave/actions

### Check Hugging Face Space:
https://huggingface.co/spaces/AhiBucket/Hand-wave

### View Build Logs:
- Click on any workflow run in GitHub Actions
- Expand the steps to see detailed logs

---

## 🛠️ Troubleshooting

### If sync fails:

1. **Check the token**:
   - Make sure `HF_TOKEN` secret is set correctly in GitHub
   - Token must have **Write** permission

2. **Check the workflow logs**:
   - Go to: https://github.com/Ahilan-Bucket/Hand-wave/actions
   - Click on the failed run
   - Read the error messages

3. **Manually trigger**:
   - Go to: https://github.com/Ahilan-Bucket/Hand-wave/actions
   - Click "Sync to Hugging Face Space"
   - Click "Run workflow"

### Common Issues:

**"Authentication failed"**
→ Token is incorrect or expired. Generate a new token and update the GitHub secret.

**"Remote already exists"**
→ This is normal and handled by the workflow (using `|| true`)

**"Permission denied"**
→ Make sure your token has **Write** permission, not just Read

---

## 🎨 Customization

### Change the branch that triggers sync:

Edit `.github/workflows/sync-to-huggingface.yml`:

```yaml
on:
  push:
    branches:
      - main        # Change this to your preferred branch
      - develop     # Or add multiple branches
```

### Add email notifications:

You can add email notifications in GitHub Actions settings:
1. Go to: https://github.com/Ahilan-Bucket/Hand-wave/settings
2. Click "Notifications"
3. Configure as desired

---

## ✅ Benefits

- ✨ **Zero manual deployment** - Just commit and push
- 🚀 **Fast updates** - Changes go live in minutes
- 🔒 **Secure** - Token stored as GitHub secret
- 📊 **Trackable** - See deployment history in GitHub Actions
- 🔄 **Reliable** - Automatic retries if deployment fails

---

## 📚 Additional Resources

- GitHub Actions Docs: https://docs.github.com/en/actions
- Hugging Face Spaces Docs: https://huggingface.co/docs/hub/spaces
- Your Workflow File: `.github/workflows/sync-to-huggingface.yml`

---

**That's it!** After the one-time setup, you'll never need to manually deploy again. Just code, commit, and push! 🎉
