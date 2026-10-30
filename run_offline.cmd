@echo off
rem Starts the VoiceChat browser demo (voice UI + live animated pipeline terminal, via run_viz.py) with all model-hub network access disabled.
rem Requires Ollama to be running (it serves the qwen2.5:3b LLM on localhost:11434).
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONUTF8=1
rem Start Ollama in the background if it is not already answering, and wait for it (up to ~30s).
powershell -NoProfile -Command "try{Invoke-WebRequest http://127.0.0.1:11434/api/version -UseBasicParsing -TimeoutSec 2|Out-Null}catch{Write-Host 'Starting Ollama...';Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden;for($i=0;$i -lt 30;$i++){try{Invoke-WebRequest http://127.0.0.1:11434/api/version -UseBasicParsing -TimeoutSec 2|Out-Null;break}catch{Start-Sleep 1}}}"
rem "run_offline.cmd open" also opens the browser once the server answers (waits up to ~4 min).
if /i "%~1"=="open" start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 120;$i++){try{Invoke-WebRequest http://127.0.0.1:8000/ -UseBasicParsing -TimeoutSec 2|Out-Null;Start-Process ('http://127.0.0.1:8000/live?t='+[DateTimeOffset]::Now.ToUnixTimeSeconds());break}catch{Start-Sleep 2}}"
".venv\Scripts\python.exe" run_viz.py
