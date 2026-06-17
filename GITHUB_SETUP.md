# GitHub Repository Setup Instructions

A curated reproducibility package is ready in `github_submission_package/`.  
Choose **Option A** (automatic) if the GitHub CLI (`gh`) is installed; otherwise use **Option B** (manual).

## Option A — Automatic creation with GitHub CLI

1. Install `gh` if you have not already: https://cli.github.com/
2. Authenticate: `gh auth login`
3. Open a PowerShell terminal in the project root and run:

   ```powershell
   .\github_submission_package\create_and_push_repo.ps1 -RepoName "YourRepoName"
   ```

   The script will:
   - create a public repository under your authenticated GitHub account,
   - push the contents of `github_submission_package/`,
   - print the final repository URL.

## Option B — Manual creation and push

1. Go to https://github.com/new and create a **public** repository (e.g. `thesis-ecm-repro`).  
   Do **not** initialize it with a README, license, or `.gitignore`; those files are already included in the package.
2. Open a terminal in `github_submission_package/`:

   ```powershell
   cd github_submission_package
   git init
   git add .
   git commit -m "Initial reproducibility package for ECM submission"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

3. Copy the final repository URL, e.g. `https://github.com/YOUR_USERNAME/YOUR_REPO_NAME`.

## Updating the manuscript with the real URL

After the repository exists, run the helper script from the project root:

```powershell
.\.venv\Scripts\python.exe replace_github_url.py "https://github.com/YOUR_USERNAME/YOUR_REPO_NAME"
```

This replaces the placeholder in the Data availability statement and regenerates the submission DOCX with the next version number.

## Files that are published

- `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`
- `config.py`
- `src/` (KAN forecasting, scenario generation, TSSP model)
- `experiments/` (sensitivity and validation scripts)
- `data/` (processed wind/solar prediction CSVs)

Raw capacity-factor data remain available upon request as stated in the manuscript's Data availability statement.
