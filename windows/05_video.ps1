$ErrorActionPreference = "Stop"
& .\.venv\Scripts\python.exe -m rl01.cli video --run-dir results/tier1/dqn_base_seed0011 --out results/tier1/cartpole_dqn.gif
& .\.venv\Scripts\python.exe -m rl01.cli video --run-dir results/tier2/dqn_base_seed0011 --out results/tier2/acrobot_dqn.gif
