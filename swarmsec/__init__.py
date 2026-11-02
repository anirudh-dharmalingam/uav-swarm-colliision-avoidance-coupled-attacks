"""swarmsec: a headless cyber-physical-security testbed for distributed UAV swarms.

Layer-1 abstract simulator (3D double-integrator swarm + higher-order CBF-QP
safety filter) used to characterise *collective safety-feasibility erosion* and
to evaluate resilient defenses. The same instrumentation (safety-feasibility
margin, QP correction, QP slack, algebraic connectivity) is reused by the
high-fidelity PX4/Gazebo Layer-2 controller.

Everything here runs inside the DroneLab research image.
"""

__version__ = "0.1.0"
