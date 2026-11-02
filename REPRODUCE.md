# Reproduction

Run commands from the repository root.

```bash
python3 tools/verify_bundle.py
```

This standard-library check verifies the packaged file hashes, experiment freeze hashes, the presence of the 252 primary flight truth traces, and the row counts of the recorded planar studies.

For numerical analysis and tests, use Python 3.12 with `requirements-l1.txt`, or build the research image from the included `Dockerfile`. The image build needs network access, significant disk space, and time; its dependencies may have changed since the original build. After installing the dependencies, run:

```bash
python3 tests/run_b_tests.py
```

Experiment specifications and selection rules are in `experiments/`. Recorded episode summaries are in `results/rarf_l1/`, and PX4 flight traces are in `results/rarf_hocbf_v1/`. Frozen analyses are in `output/research/`. The primary flight population and attempt-selection mapping are in `output/research/rarf_hocbf_v1/confirm_primary_population/`.

The planar simulator is an approximation calibrated against PX4 software-in-the-loop records. PX4 flights are not bit-reproducible. In the planar simulator, some MR-HOCBF reruns differ from recorded outcomes because the solver cache is shared within a worker process; see the experiment plans and source code for the setup. The package does not include all per-agent PX4 telemetry and all per-episode planar traces.
