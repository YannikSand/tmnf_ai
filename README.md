TMNF AI — Teaching a Car to Drive (With Mixed Results)

Welcome to TMNF AI, a project exploring whether a machine can learn to drive in TrackMania Nations Forever without immediately steering into the nearest wall.

Results so far: inconsistent, but promising.

Overview

This project implements an AI agent for TrackMania Nations Forever (TMNF) using machine learning techniques. The goal is to train a model that can navigate tracks, improve lap times, and gradually make fewer catastrophic decisions.

It’s a mix of experimentation, reinforcement learning, and watching a program slowly figure out that walls are not shortcuts.

How It Works

The system follows a standard reinforcement learning loop:

Observe the environment (game state, visuals, telemetry, etc.)
Take an action (steering, acceleration, braking)
Receive feedback (reward or penalty)
Repeat many times until something resembling skill emerges

Early behavior may look less like racing and more like interpretive chaos. This is expected.

Features
AI agent capable of interacting with TMNF
Training pipeline for iterative improvement
Modular structure for experimentation
Model saving and evaluation support
Project Structure
tmnf_ai/
│── data/              # Training data and logs
│── models/            # Saved models
│── training/          # Training logic
│── inference/         # Running trained agents
│── utils/             # Utility functions
│── main.py            # Entry point
│── README.md
Installation
1. Clone the repository
git clone https://github.com/YannikSand/tmnf_ai.git
cd tmnf_ai
2. Install dependencies
pip install -r requirements.txt
3. Install TrackMania Nations Forever

The AI will need the game to run. There is no simulation fallback here.

Usage
Train the model
python train.py
Run the trained agent
python run.py
Evaluate performance
python evaluate.py
Training Notes
Initial performance will be poor
The agent may repeatedly fail in creative ways
Improvement is gradual and depends heavily on tuning

Patience is not optional.
