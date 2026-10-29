@echo off
setlocal
cd /d "%~dp0"

echo Starting NexusBI dashboard...
echo Open http://localhost:8501 in your browser.
echo Close this window to stop the dashboard.

python -m streamlit run dashboard\app.py --server.address localhost --server.port 8501
if errorlevel 1 (
    echo.
    echo Dashboard failed to start. Check the error above.
    pause
)
endlocal
