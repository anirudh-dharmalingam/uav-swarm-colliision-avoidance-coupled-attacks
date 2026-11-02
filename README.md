# Control Authority and Information Uncertainty in UAV Swarm Avoidance under Coupled Attacks

Research code and experimental records for distributed UAV collision avoidance under coupled pursuit and state-report manipulation.

- `swarmsec/` and `controllers/` contain avoidance, control and network implementations.
- `experiments/` contains study designs, specifications, freeze records and deviations.
- `results/` contains episode-level simulator results and packaged PX4 software-in-the-loop flight records.
- `output/research/` contains recorded analyses and calibration outputs.
- `scripts/`, `tests/`, `config/` and `worlds/` contain analysis, execution and validation resources.
- `Dockerfile` defines the research environment; the built image is not included.

Run `python3 tools/verify_bundle.py` to check the file manifest, frozen experiment hashes and record completeness. See [REPRODUCE.md](REPRODUCE.md) for environment and analysis details.

The research snapshot included uncommitted source files. Study-specific freeze records identify the recorded code and deviations. PX4 experiments are software-in-the-loop, and broad studies use a calibrated planar simulator.
