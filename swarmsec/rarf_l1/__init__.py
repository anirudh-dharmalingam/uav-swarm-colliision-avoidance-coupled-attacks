"""Layer-1 fast simulator for the RARF / coupled-attack avoidance study (transactions programme, B-F0).

It runs the UNMODIFIED per-UAV PX4 agent classes (controllers/swarm/cs_agent_node_v4.CsAgentV4 and its parents: nominal law, attack
terms, broadcast, refresh requests, plant-time scheduling, every avoidance policy) inside a stub ROS 2 / PX4 layer, with
  * a calibrated planar plant (second-order acceleration lag, fitted on rarf_plant_time_v3 PX4 logs: plant_fit.json),
  * ego-estimate errors as Ornstein-Uhlenbeck processes (same fit),
  * the same application-level network (swarmsec.fanet.Fanet / KeyedFanet, as fanet_bridge.py),
  * the GCS start / land sequencing of gcs_coordinator.py,
and writes PX4-format episode directories so scripts/coupled_safety/metrics.py scores Layer 1 and Layer 2 identically.
Not modelled: PX4 failsafes, rotor contact dynamics (contact < 0.77 m is recorded and both vehicles are stopped as crashed), downwash,
altitude-loop coupling, OS timing (no deadline misses).
"""
