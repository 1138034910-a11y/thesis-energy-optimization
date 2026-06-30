#Requires -Version 7
<#
.SYNOPSIS
    Create a public GitHub repository and push the github_submission_package contents.
.PARAMETER RepoName
    Name of the repository to create.
.PARAMETER Owner
    GitHub owner/username. If omitted, the authenticated gh user is used.
.PARAMETER Public
    Make the repository public (default).
.PARAMETER Private
    Make the repository private.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoName,

    [string]$Owner = "",

    [switch]$Public = $true,

    [switch]$Private
)

$ErrorActionPreference = "Stop"

# Check prerequisites
$gh = Get-Command gh -ErrorAction SilentlyContinue
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $gh) {
    Write-Error "GitHub CLI (gh) is not installed. Install it from https://cli.github.com/ or use Option B in GITHUB_SETUP.md."
}
if (-not $git) {
    Write-Error "Git is not installed. Install it from https://git-scm.com/."
}

# Resolve paths
$root = Split-Path -Parent $PSScriptRoot
$pkg = Join-Path $root "github_submission_package"
if (-not (Test-Path $pkg)) {
    Write-Error "Could not find github_submission_package at $pkg"
}

$visibility = if ($Private) { "private" } else { "public" }

# Create repo
Write-Host "Creating GitHub repository '$RepoName' ($visibility)..."
if ($Owner) {
    gh repo create "$Owner/$RepoName" --$visibility --source "$pkg" --remote=origin --push
} else {
    gh repo create "$RepoName" --$visibility --source "$pkg" --remote=origin --push
}

# Build URL
if ($Owner) {
    $url = "https://github.com/$Owner/$RepoName"
} else {
    $user = gh api user -q '.login'
    $url = "https://github.com/$user/$RepoName"
}

Write-Host "Repository created: $url" -ForegroundColor Green
Write-Host "Repository URL: $url" -ForegroundColor Green
Write-Host "Next step: run the following command from the project root to update the manuscript with this URL:" -ForegroundColor Cyan
Write-Host ".\.venv\Scripts\python.exe replace_github_url.py `"$url`"" -ForegroundColor Cyan
