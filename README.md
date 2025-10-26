A-CIS (Academic Cache Intelligent Simulator)

This project provides a minimal DQN (Deep Q-Network) based cache replacement simulator.

Structure:
- main.py: runner script to train and evaluate the agent.
- config.py: configuration parameters (cache, RL, training).
- agent/rl_agent.py: DQN agent implementation using PyTorch.
- environment/cache_env.py: set-associative cache environment.
- utils/workload_loader.py: simple trace loader.
- utils/plotter.py: plotting utilities.
- traces/sample_trace.txt: tiny sample trace for quick testing.

Quick start:
1. Create a Python environment (Python 3.8+ recommended).
2. Install dependencies:
   pip install -r requirements.txt
3. Run the simulator:
   python main.py

Notes:
- The DQN here is intentionally minimal for clarity and portability.
- Adjust hyperparameters in `config.py` for longer experiments.
