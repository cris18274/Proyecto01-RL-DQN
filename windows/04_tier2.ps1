$ErrorActionPreference = "Stop"
& .\.venv\Scripts\python.exe -m rl01.cli tier2 --environment Acrobot-v1 --seeds 11,22,33 --steps 100000 --eval-every 10000
