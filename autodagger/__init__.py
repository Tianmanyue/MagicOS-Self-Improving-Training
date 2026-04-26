"""
AutoDAgger: Fully Automated DAgger in Simulation.

Implements a closed-loop self-improvement pipeline:
  train policy → evaluate → collect failures → VLM stage annotation →
  detect failure stage → expert recovery demos → retrain → re-evaluate

Core contribution: scripted policy as expert + VLM automatic stage discovery
+ progress model for failure detection = fully automated DAgger without
any human intervention.
"""

__version__ = "0.1.0"
