@echo off
cd /d "%~dp0"
".venv312\Scripts\streamlit.exe" run dashboard.py --server.headless true
