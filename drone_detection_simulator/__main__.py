"""
Main entry point for running the drone detection simulator as a module.

This allows the simulator to be run with:
    python -m drone_detection_simulator
"""

from .cli import main

if __name__ == "__main__":
    exit(main())