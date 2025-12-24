@echo off
setlocal
cd /d "%~dp0"
echo "正在启动 SHARP Web..."
if exist ".venv\\Scripts\\activate.bat" (
  call ".venv\\Scripts\\activate.bat"
)
python app\\server.py
pause
