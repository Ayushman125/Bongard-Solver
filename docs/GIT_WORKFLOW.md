# Git Workflow Guide: Publishing to GitHub

This guide provides step-by-step instructions for publishing your research to GitHub with proper version control practices.

---

## Prerequisites

1. **Install Git**: Download from [git-scm.com](https://git-scm.com/)
2. **Create GitHub Account**: Sign up at [github.com](https://github.com/)
3. **Configure Git** (first-time setup):

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Verify configuration:**
```powershell
git config --list
```

---

## Initialization

### Step 1: Initialize Repository

Navigate to your project directory and initialize Git:

```powershell
cd c:\Users\HP\AI_Projects\BongordSolver
git init
```

**Output:**
```
Initialized empty Git repository in c:/Users/HP/AI_Projects/BongordSolver/.git/
```

### Step 2: Create `.gitignore`

Prevent tracking of unnecessary files:

```powershell
@"
# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv

# PyTorch models
*.pth
*.pt
pretrained/

# Logs & metrics
logs/
metrics/
*.log

# Data (too large)
data/raw/
data/processed/
cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
"@ | Out-File -FilePath .gitignore -Encoding utf8
```

**Verify:**
```powershell
cat .gitignore
```

---

## Initial Commit

### Step 3: Stage Files

Add all project files to staging area:

```powershell
git add .
```

**Check status:**
```powershell
git status
```

**Expected output:**
```
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   .gitignore
        new file:   README.md
        new file:   bayes_dual_system/__init__.py
        new file:   bayes_dual_system/types.py
        new file:   bayes_dual_system/data.py
        ...
```

### Step 4: Create Initial Commit

```powershell
git commit -m "Initial commit: Hybrid Dual-System Solver for BONGARD-LOGO

- Integrated System 1 (ResNet-15 neural) + System 2 (Bayesian rule induction)
- Self-supervised pretraining (100 epochs rotation prediction)
- Meta-learned arbitration with weight cap innovation (max_w2=0.95)
- Achieved SOTA: 100%/92.7%/73.0%/73.4% (test_ff/bd/hd_comb/hd_novel)
- Comprehensive documentation (README, ARCHITECTURE, RESULTS, THEORY)
- Ablation study scripts for reproducibility
- Outperforms Meta-Baseline-PS by +31.8% on test_ff"
```

**Verify commit:**
```powershell
git log --oneline
```

---

## Create GitHub Repository

### Step 5: Create Remote Repository

**Option A: Via GitHub Web Interface**

1. Go to https://github.com/new
2. **Repository name**: `BongardSolver` (or your preferred name)
3. **Description**: "Hybrid Dual-System Solver for BONGARD-LOGO: Integrates neural pattern recognition (System 1) with Bayesian rule induction (System 2) via meta-learned arbitration. Achieves new SOTA on freeform split (100%)."
4. **Visibility**: 
   - ✅ **Public** (recommended for research sharing)
   - ⬜ Private (if you prefer restricted access)
5. **Initialize**: 
   - ⬜ **DO NOT** add README (you already have one)
   - ⬜ **DO NOT** add .gitignore (you already have one)
   - ⬜ **DO NOT** choose license yet (can add later)
6. Click **"Create repository"**

**Option B: Via GitHub CLI** (if installed):

```powershell
gh repo create BongardSolver --public --source=. --remote=origin --description "Hybrid Dual-System Solver for BONGARD-LOGO"
```

---

## Connect to Remote

### Step 6: Add Remote Origin

Copy the URL from GitHub:

```powershell
git remote add origin https://github.com/Ayushman125/Bongard-Solver.git
```

**Verify:**
```powershell
git remote -v
```

**Output:**
```
origin  https://github.com/Ayushman125/Bongard-Solver.git (fetch)
origin  https://github.com/Ayushman125/Bongard-Solver.git (push)
```

---

## Push to GitHub

### Step 7: Rename Branch to `main`

GitHub now uses `main` as default (not `master`):

```powershell
git branch -M main
```

### Step 8: Push Initial Commit

```powershell
git push -u origin main
```

**Authentication prompt** (first-time):
- **Username**: your GitHub username
- **Password**: Use **Personal Access Token** (PAT), not account password
  - Generate PAT: https://github.com/settings/tokens
  - Scopes: `repo` (full control)
  - Copy and paste token as password

**Expected output:**
```
Enumerating objects: 42, done.
Counting objects: 100% (42/42), done.
Delta compression using up to 8 threads
Compressing objects: 100% (35/35), done.
Writing objects: 100% (42/42), 52.34 KiB | 5.23 MiB/s, done.
Total 42 (delta 12), reused 0 (delta 0), pack-reused 0
To https://github.com/Ayushman125/Bongard-Solver.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

---

## Subsequent Commits

### Making Changes

After editing files, follow this workflow:

```powershell
# 1. Check which files changed
git status

# 2. Review specific changes
git diff bayes_dual_system/dual_system_hybrid.py

# 3. Stage changes
git add bayes_dual_system/dual_system_hybrid.py
# Or stage all changes:
git add .

# 4. Commit with descriptive message
git commit -m "Fix: Prevent weight saturation in arbitrator

- Cap system2_weight at 0.95 instead of 0.995
- Improves test_bd accuracy from 89.3% to 92.7%
- Prevents S1 marginalization (S1_weight >= 0.05 always)"

# 5. Push to GitHub
git push
```

---

## Commit Message Best Practices

### Structure

```
<type>(<scope>): <short summary>

<detailed description>

<footer: references, metrics>
```

### Types

- **feat**: New feature (e.g., `feat(system2): Add compositional rule generation`)
- **fix**: Bug fix (e.g., `fix(system1): Reset classifier head before episode`)
- **docs**: Documentation (e.g., `docs(README): Add ablation study section`)
- **perf**: Performance improvement (e.g., `perf(features): Cache extracted features`)
- **refactor**: Code restructuring (e.g., `refactor(data): Simplify episode loading`)
- **test**: Add tests (e.g., `test(system1): Add pretraining unit tests`)
- **chore**: Maintenance (e.g., `chore: Update dependencies`)

### Examples

**Good ✅:**
```
feat(pretraining): Add self-supervised rotation prediction

- Pretrain ResNet backbone on 111,600 images (100 epochs)
- 4-way rotation classification task {0°, 90°, 180°, 270°}
- Improves test_ff accuracy from 87.2% to 100.0%
- Reduces fine-tuning steps from 50 to 10

Closes #12
```

**Bad ❌:**
```
update stuff
```

---

## Branching Strategy

### Feature Branches

For major experimental changes, use feature branches:

```powershell
# Create and switch to new branch
git checkout -b experiment/auto-cap-tuning

# Make changes and commit
git add .
git commit -m "feat(arbitration): Implement auto-cap validation tuning"

# Push branch to GitHub
git push -u origin experiment/auto-cap-tuning

# After testing, merge to main
git checkout main
git merge experiment/auto-cap-tuning
git push
```

### Branch Naming Conventions

- `experiment/*`: Experimental features (e.g., `experiment/triple-system`)
- `fix/*`: Bug fixes (e.g., `fix/head-reset`)
- `docs/*`: Documentation updates (e.g., `docs/add-theory-md`)
- `perf/*`: Performance optimizations (e.g., `perf/gpu-acceleration`)

---

## Tagging Releases

### Semantic Versioning

Mark important milestones with tags:

```powershell
# Tag current commit
git tag -a v1.0.0 -m "v1.0.0: Initial SOTA release

- Achieved 100%/92.7%/73.0%/73.4% on BONGARD-LOGO
- Hybrid dual-system architecture
- Self-supervised pretraining
- Meta-learned arbitration with cap"

# Push tag to GitHub
git push origin v1.0.0
```

**Version format**: `v<major>.<minor>.<patch>`
- **Major**: Breaking changes (e.g., v2.0.0)
- **Minor**: New features (e.g., v1.1.0)
- **Patch**: Bug fixes (e.g., v1.0.1)

### Viewing Tags

```powershell
git tag
git show v1.0.0
```

---

## Collaboration

### Pull Requests

If collaborating with others:

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Commit** changes
5. **Push** to your fork
6. **Open** a Pull Request on GitHub
7. **Review** and merge

### Code Review Workflow

```powershell
# Reviewer: Check out PR branch
git fetch origin pull/42/head:pr-42
git checkout pr-42

# Test changes
python run_experiment.py --split bd --cap 0.95

# If approved, merge on GitHub
```

---

## GitHub Repository Setup

### Add README Features

**Badges** (add to top of README.md):

```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25%20(test__ff)-brightgreen)
```

### Add License

**MIT License** (create `LICENSE` file):

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Add to Git:**
```powershell
git add LICENSE
git commit -m "docs: Add MIT license"
git push
```

### Enable GitHub Features

1. **Actions**: Automated testing (`.github/workflows/test.yml`)
2. **Issues**: Bug tracking and feature requests
3. **Wiki**: Extended documentation
4. **Discussions**: Community Q&A
5. **Releases**: Binary distributions with changelogs

---

## Troubleshooting

### Authentication Failed

**Problem**: HTTPS push requires PAT (not password)

**Solution**: Generate Personal Access Token
```powershell
# Navigate to: https://github.com/settings/tokens
# Click "Generate new token (classic)"
# Select scopes: repo (full control)
# Copy token and use as password
```

**Alternative**: Use SSH instead of HTTPS
```powershell
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to ssh-agent
ssh-add ~/.ssh/id_ed25519

# Add public key to GitHub (Settings > SSH Keys)
cat ~/.ssh/id_ed25519.pub

# Change remote URL
git remote set-url origin git@github.com:Ayushman125/Bongard-Solver.git
```

### Merge Conflicts

**Problem**: Diverged branches with conflicting changes

**Solution**:
```powershell
# Pull remote changes
git pull origin main

# Git will mark conflicts in files:
# <<<<<<< HEAD
# Your changes
# =======
# Remote changes
# >>>>>>> origin/main

# Manually resolve, then:
git add resolved_file.py
git commit -m "Merge: Resolve conflicts in resolved_file.py"
git push
```

### Large Files

**Problem**: GitHub rejects files >100MB

**Solution**: Use Git LFS (Large File Storage)
```powershell
# Install Git LFS
git lfs install

# Track large files
git lfs track "pretrained/*.pth"
git lfs track "data/raw/**"

# Add .gitattributes
git add .gitattributes
git commit -m "chore: Add Git LFS for model weights"
git push
```

### Undo Last Commit (Not Pushed)

```powershell
# Keep changes, undo commit
git reset --soft HEAD~1

# Discard changes, undo commit
git reset --hard HEAD~1
```

### Undo Pushed Commit

```powershell
# Revert creates new commit that undoes previous one
git revert HEAD
git push
```

---

## Complete Workflow Summary

```powershell
# === ONE-TIME SETUP ===
cd c:\Users\HP\AI_Projects\BongordSolver
git init
# Create .gitignore (see Step 2)
git add .
git commit -m "Initial commit: ..."
git remote add origin https://github.com/Ayushman125/Bongard-Solver.git
git branch -M main
git push -u origin main

# === DAILY WORKFLOW ===
# 1. Make changes
code bayes_dual_system/dual_system_hybrid.py

# 2. Check status
git status
git diff

# 3. Stage and commit
git add .
git commit -m "fix(arbitration): Cap system2_weight at 0.95"

# 4. Push to GitHub
git push

# === TAGGING RELEASES ===
git tag -a v1.0.0 -m "v1.0.0: Initial SOTA release"
git push origin v1.0.0

# === BRANCHING FOR EXPERIMENTS ===
git checkout -b experiment/new-feature
# ... make changes ...
git add .
git commit -m "feat: Add new feature"
git push -u origin experiment/new-feature
# ... test, then merge on GitHub ...
git checkout main
git pull
```

---

## Recommended GitHub Repository Description

**About Section:**

> Hybrid Dual-System Solver for BONGARD-LOGO: Integrates neural pattern recognition (System 1, ResNet-15) with Bayesian rule induction (System 2) via meta-learned arbitration. Achieves new SOTA on freeform concepts (100% accuracy). Implements self-supervised pretraining, episodic meta-learning, and weight cap innovation.

**Topics** (GitHub tags):
```
few-shot-learning, meta-learning, concept-learning, 
dual-process-theory, bayesian-inference, bongard-problems,
pytorch, self-supervised-learning, hybrid-ai
```

**Website**: Link to arXiv paper (if published) or project page

---

## Next Steps After Publishing

1. **Add Citation**:
   - Create `CITATION.cff` (citation metadata)
   - Or add BibTeX to README

2. **Documentation Site**:
   - Use GitHub Pages for HTML docs
   - Or link to ReadTheDocs

3. **CI/CD**:
   - Add `.github/workflows/test.yml` for automated testing
   - Run linters (black, flake8, mypy)

4. **Community**:
   - Add CONTRIBUTING.md guidelines
   - Create issue templates
   - Respond to GitHub Discussions

5. **Publication**:
   - Submit to arXiv (if research paper)
   - Share on Reddit (r/MachineLearning)
   - Tweet with #ML tag

---

## Summary Checklist

Before pushing to GitHub, verify:

- [x] `.gitignore` configured (excludes data, models, logs)
- [x] README.md complete (usage, results, citation)
- [x] LICENSE file added (MIT recommended)
- [x] Code documented (docstrings, comments)
- [x] Scripts tested and working
- [x] Results reproducible
- [x] File paths use relative paths (not hardcoded)
- [x] No sensitive data (API keys, passwords)
- [x] Repository description and topics set
- [x] First commit message is descriptive

**You're ready to publish!** 🚀

