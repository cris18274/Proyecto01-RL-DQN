$ErrorActionPreference = "Stop"
& .\.venv\Scripts\python.exe -m rl01.cli tier1 --seeds 11,22,33 --steps 80000 --eval-every 10000
