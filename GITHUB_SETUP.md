# GitHub Repository Setup Instructions

A curated reproducibility package is ready in `github_submission_package/`.  
Choose **Option A** (automatic) if the GitHub CLI (`gh`) is installed; otherwise use **Option B** (manual).

**Repository URL:** https://github.com/1138034910-a11y/thesis-energy-optimization

## Option A — Automatic creation with GitHub CLI

1. Install `gh` if you have not already: https://cli.github.com/
2. Authenticate: `gh auth login`
3. Open a PowerShell terminal in the project root and run:

   ```powershell
   .\github_submission_package\create_and_push_repo.ps1 -RepoName "thesis-energy-optimization"
   ```

   The script will:
   - create a public repository under your authenticated GitHub account,
   - push the contents of `github_submission_package/`,
   - print the final repository URL.

## Option B — Manual creation and push

1. Go to https://github.com/new and create a **public** repository named `thesis-energy-optimization`.  
   Do **not** initialize it with a README, license, or `.gitignore`; those files are already included in the package.
2. Open a terminal in `github_submission_package/`:

   ```powershell
   cd github_submission_package
   git init
   git add .
   git commit -m "JES submission reproducibility package"
   git branch -M main
   git remote add origin https://github.com/1138034910-a11y/thesis-energy-optimization.git
   git push -u origin main
   ```

3. Verify the repository is live at: https://github.com/1138034910-a11y/thesis-energy-optimization

## Updating an existing repository

If the repository already exists and you only need to push updates:

```powershell
cd github_submission_package
git add .
git commit -m "Update JES reproducibility package"
git push origin main
```

## Files that are published

- `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`
- `config.py`
- `src/` (KAN forecasting, scenario generation, TSSP model)
- `experiments/` (sensitivity and validation scripts)
- `scripts/` (figure generation and post-processing)
- `data/` (input datasets: wind/solar profiles and KAN prediction results; see `data/README.md`)

The following are **not** included in this repository and will be generated locally when the code is run:

- Solver logs and intermediate CSV/JSON result files
- Manuscript figures and tables

Any additional raw data not included here remain available upon request as stated in the manuscript's Data availability statement.
