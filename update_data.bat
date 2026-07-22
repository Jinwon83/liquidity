@echo off
cd /d "%~dp0"
if not exist ".venv312\Scripts\python.exe" (
  echo [ERR] .venv312 ??. ?? setup ??.
  exit /b 1
)
echo === 1) collect ===
".venv312\Scripts\python.exe" run_stage12.py
if errorlevel 1 exit /b 1
echo === 2) analysis ===
".venv312\Scripts\python.exe" correlation_analysis.py
if errorlevel 1 exit /b 1
echo === 3) export pages json ===
".venv312\Scripts\python.exe" scripts\export_pages_data.py
if errorlevel 1 exit /b 1
echo === 4) git commit --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>"/push ===
git add data docs/data
git diff --staged --quiet
if errorlevel 1 (
  git commit --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>" -m "chore: update dashboard data"
  git push
) else (
  echo no data changes to commit
)
echo DONE
pause
