"""
Drone Detection Simulator Package

A Python-based drone detection simulator that generates realistic camera-based 
drone detections using proper geometric calculations and publishes them via MQTT.
"""

__version__ = "0.1.0"
__author__ = "Drone Detection Simulator"

from .config import SimulatorConfig
from .camera import CameraModel
from .motion import MotionGenerator
from .detection import DetectionGenerator
from .projection import world_to_pixel_projection
from .mqtt_publisher import MQTTPublisher
from .timing import TimingController, SimulationLoop, FrameInfo, format_timestamp_utc
from .simulator import DroneSimulator, main
from .cli import main as cli_main

__all__ = [
    "SimulatorConfig", 
    "CameraModel", 
    "MotionGenerator", 
    "DetectionGenerator",
    "world_to_pixel_projection",
    "MQTTPublisher",
    "TimingController",
    "SimulationLoop", 
    "FrameInfo",
    "format_timestamp_utc",
    "DroneSimulator",
    "main",
    "cli_main"
]