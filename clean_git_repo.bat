:: filepath: c:\Users\engin\projects\samGui\clean_git_repo.bat
@echo off
echo Cleaning .pth files from git history...

:: Check if git-filter-repo is installed
python -m git_filter_repo --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo git-filter-repo is not installed. Installing it now...
    pip install git-filter-repo
)

:: Run git-filter-repo to remove .pth files
python -m git_filter_repo --path-glob "*.pth" --invert-paths

:: Optimize the repository
echo Optimizing the repository...
git gc --prune=now --aggressive

:: Show the repository size
echo Repository size after cleanup:
git count-objects -vH

echo Cleanup complete!
pause