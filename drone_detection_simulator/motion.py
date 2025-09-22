"""
Motion generation for drone detection simulator.

This module provides physics-based ENU path planning for smooth drone motion
with configurable span, speed, and acceleration constraints.
"""

import numpy as np
from typing import List, Tuple
from .config import SimulatorConfig


class MotionGenerator:
    """
    Generates realistic drone flight paths with physics-based constraints.
    
    Creates smooth left-to-right motion in ENU coordinates with configurable
    altitude, span, speed, and maximum lateral acceleration limits to prevent
    unrealistic teleporting motion.
    """
    
    def __init__(self, config: SimulatorConfig):
        """
        Initialize motion generator with configuration parameters.
        
        Args:
            config: SimulatorConfig containing motion parameters
        """
        self.config = config
        self.paths = self._generate_drone_paths()
        self.total_frames = int(config.duration_s * config.fps)
        self.frame_time_s = 1.0 / config.fps
    
    def _generate_drone_paths(self) -> List[List[np.ndarray]]:
        """
        Generate smooth ENU paths for all drones.
        
        Creates physics-based trajectories that respect acceleration constraints
        and provide smooth left-to-right motion across the camera's field of view.
        
        Returns:
            List of paths, where each path is a list of ENU position vectors
        """
        paths = []
        
        for drone_idx in range(self.config.num_drones):
            path = self._generate_single_drone_path(drone_idx)
            paths.append(path)
        
        return paths
    
    def _generate_single_drone_path(self, drone_idx: int) -> List[np.ndarray]:
        """
        Generate a single drone's path with smooth motion.
        
        Args:
            drone_idx: Index of the drone (for staggering multiple drones)
            
        Returns:
            List of ENU position vectors for this drone
        """
        total_frames = int(self.config.duration_s * self.config.fps)
        path = []
        
        # Stagger start times for multiple drones
        start_delay = drone_idx * (self.config.duration_s / max(self.config.num_drones, 1)) * 0.3
        
        # Generate path for each frame
        for frame in range(total_frames):
            time_s = frame / self.config.fps
            adjusted_time = time_s - start_delay
            
            position = self._compute_position_at_time(adjusted_time, drone_idx)
            path.append(position)
        
        return path
    
    def _compute_position_at_time(self, time_s: float, drone_idx: int) -> np.ndarray:
        """
        Compute ENU position at a specific time with smooth acceleration.
        
        Args:
            time_s: Time in seconds (can be negative for pre-start)
            drone_idx: Drone index for slight variations
            
        Returns:
            ENU position vector [east, north, up] in meters
        """
        # Start position (left side of span)
        start_east = -self.config.path_span_m / 2.0
        
        # Add slight north offset for multiple drones
        north_offset = (drone_idx - (self.config.num_drones - 1) / 2.0) * 2.0
        
        # Altitude with slight variation per drone
        altitude = self.config.path_altitude_agl_m + drone_idx * 0.5
        
        if time_s <= 0:
            # Before motion starts
            east_pos = start_east
        else:
            # Calculate smooth motion with acceleration constraints
            east_pos = self._compute_smooth_east_position(time_s, start_east)
        
        return np.array([east_pos, north_offset, altitude])
    
    def _compute_smooth_east_position(self, time_s: float, start_east: float) -> float:
        """
        Compute east position with smooth acceleration profile.
        
        Uses either triangular or trapezoidal velocity profile based on the
        configured parameters and available time.
        
        Args:
            time_s: Time since motion start
            start_east: Starting east position
            
        Returns:
            East position in meters
        """
        if time_s <= 0:
            return start_east
        
        total_distance = self.config.path_span_m
        max_accel = self.config.max_lateral_accel_mps2
        max_speed = self.config.speed_mps
        simulation_duration = self.config.duration_s
        
        # The motion should span most of the simulation duration
        # Leave some buffer at the end for the drone to be visible
        motion_duration = simulation_duration * 0.8  # Use 80% of simulation time
        
        # Calculate the maximum distance we can cover with the given constraints
        # If we can't cover the full path_span_m, we'll cover what we can
        max_distance_possible = self._calculate_max_distance_in_time(motion_duration, max_speed, max_accel)
        actual_distance = min(total_distance, max_distance_possible)
        
        # Calculate motion profile parameters
        accel_time_to_max_speed = max_speed / max_accel
        accel_distance_to_max_speed = 0.5 * max_accel * accel_time_to_max_speed**2
        
        # Check if we can reach max speed given the distance constraint
        if 2 * accel_distance_to_max_speed >= actual_distance:
            # Triangular profile - can't reach max speed due to distance
            peak_speed = np.sqrt(max_accel * actual_distance)
            accel_time = peak_speed / max_accel
            total_motion_time = 2 * accel_time
            
            # Check if this fits in available time
            if total_motion_time > motion_duration:
                # Scale down acceleration to fit in time
                accel_time = motion_duration / 2
                actual_accel = actual_distance / (accel_time**2)
                peak_speed = actual_accel * accel_time
                total_motion_time = motion_duration
            else:
                actual_accel = max_accel
            
            # Triangular profile computation
            if time_s <= accel_time:
                # Accelerating phase
                distance = 0.5 * actual_accel * time_s**2
            elif time_s <= total_motion_time:
                # Decelerating phase
                decel_time = time_s - accel_time
                distance_at_peak = 0.5 * actual_accel * accel_time**2
                decel_distance = peak_speed * decel_time - 0.5 * actual_accel * decel_time**2
                distance = distance_at_peak + decel_distance
            else:
                # Motion complete
                distance = actual_distance
                
        else:
            # Trapezoidal profile - can reach max speed
            const_distance = actual_distance - 2 * accel_distance_to_max_speed
            const_time = const_distance / max_speed
            total_motion_time = 2 * accel_time_to_max_speed + const_time
            
            # Check if this fits in available time
            if total_motion_time > motion_duration:
                # Need to reduce max speed to fit in time
                # Solve for speed: motion_duration = 2*speed/accel + (distance - 2*0.5*accel*(speed/accel)^2)/speed
                # Simplify: motion_duration = 2*speed/accel + distance/speed - speed/accel
                # motion_duration = speed/accel + distance/speed
                # This is a quadratic equation in speed
                a = 1.0 / max_accel
                b = -motion_duration
                c = actual_distance
                discriminant = b**2 - 4*a*c
                
                if discriminant >= 0:
                    # Use the smaller positive root
                    reduced_speed = (-b - np.sqrt(discriminant)) / (2*a)
                    reduced_speed = min(reduced_speed, max_speed)
                else:
                    # Fall back to triangular profile
                    accel_time = motion_duration / 2
                    actual_accel = actual_distance / (accel_time**2)
                    peak_speed = actual_accel * accel_time
                    
                    if time_s <= accel_time:
                        distance = 0.5 * actual_accel * time_s**2
                    elif time_s <= motion_duration:
                        decel_time = time_s - accel_time
                        distance_at_peak = 0.5 * actual_accel * accel_time**2
                        decel_distance = peak_speed * decel_time - 0.5 * actual_accel * decel_time**2
                        distance = distance_at_peak + decel_distance
                    else:
                        distance = actual_distance
                    
                    distance = max(0, min(distance, actual_distance))
                    return start_east + distance
                
                # Use reduced speed trapezoidal profile
                accel_time = reduced_speed / max_accel
                accel_distance = 0.5 * max_accel * accel_time**2
                const_distance = actual_distance - 2 * accel_distance
                const_time = const_distance / reduced_speed
                decel_start_time = accel_time + const_time
                total_motion_time = decel_start_time + accel_time
                
                if time_s <= accel_time:
                    # Accelerating phase
                    distance = 0.5 * max_accel * time_s**2
                elif time_s <= decel_start_time:
                    # Constant speed phase
                    const_phase_time = time_s - accel_time
                    distance = accel_distance + reduced_speed * const_phase_time
                elif time_s <= total_motion_time:
                    # Decelerating phase
                    decel_time = time_s - decel_start_time
                    distance = (accel_distance + const_distance + 
                               reduced_speed * decel_time - 
                               0.5 * max_accel * decel_time**2)
                else:
                    # Motion complete
                    distance = actual_distance
            else:
                # Normal trapezoidal profile fits in time
                decel_start_time = accel_time_to_max_speed + const_time
                
                if time_s <= accel_time_to_max_speed:
                    # Accelerating phase
                    distance = 0.5 * max_accel * time_s**2
                elif time_s <= decel_start_time:
                    # Constant speed phase
                    const_phase_time = time_s - accel_time_to_max_speed
                    distance = accel_distance_to_max_speed + max_speed * const_phase_time
                elif time_s <= total_motion_time:
                    # Decelerating phase
                    decel_time = time_s - decel_start_time
                    distance = (accel_distance_to_max_speed + const_distance + 
                               max_speed * decel_time - 
                               0.5 * max_accel * decel_time**2)
                else:
                    # Motion complete
                    distance = actual_distance
        
        # Clamp to valid range
        distance = max(0, min(distance, actual_distance))
        
        return start_east + distance
    
    def _calculate_max_distance_in_time(self, time_available: float, max_speed: float, max_accel: float) -> float:
        """
        Calculate the maximum distance that can be covered in the given time
        while respecting speed and acceleration constraints.
        
        Args:
            time_available: Available time for motion
            max_speed: Maximum allowed speed
            max_accel: Maximum allowed acceleration
            
        Returns:
            Maximum distance that can be covered
        """
        # Time to reach max speed
        accel_time = max_speed / max_accel
        
        if 2 * accel_time >= time_available:
            # Triangular profile - can't reach max speed in available time
            peak_time = time_available / 2
            peak_speed = max_accel * peak_time
            distance = peak_speed * peak_time  # Area of triangle
        else:
            # Trapezoidal profile - can reach max speed
            const_time = time_available - 2 * accel_time
            accel_distance = 0.5 * max_accel * accel_time**2
            const_distance = max_speed * const_time
            distance = 2 * accel_distance + const_distance
        
        return distance
    
    def get_positions_at_time(self, time_s: float) -> List[np.ndarray]:
        """
        Get current ENU positions for all drones at given time.
        
        Args:
            time_s: Current simulation time in seconds
            
        Returns:
            List of ENU position vectors for all drones
        """
        frame_idx = int(time_s * self.config.fps)
        frame_idx = max(0, min(frame_idx, len(self.paths[0]) - 1))
        
        positions = []
        for drone_idx in range(self.config.num_drones):
            if frame_idx < len(self.paths[drone_idx]):
                positions.append(self.paths[drone_idx][frame_idx])
            else:
                # Use last position if beyond path length
                positions.append(self.paths[drone_idx][-1])
        
        return positions
    
    def get_positions_at_frame(self, frame_idx: int) -> List[np.ndarray]:
        """
        Get ENU positions for all drones at a specific frame.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            List of ENU position vectors for all drones
        """
        frame_idx = max(0, min(frame_idx, len(self.paths[0]) - 1))
        
        positions = []
        for drone_idx in range(self.config.num_drones):
            if frame_idx < len(self.paths[drone_idx]):
                positions.append(self.paths[drone_idx][frame_idx])
            else:
                positions.append(self.paths[drone_idx][-1])
        
        return positions
    
    def verify_acceleration_constraints(self) -> Tuple[bool, float]:
        """
        Verify that generated paths respect acceleration constraints.
        
        Returns:
            Tuple of (constraints_satisfied, max_acceleration_found)
        """
        max_accel_found = 0.0
        
        for path in self.paths:
            if len(path) < 3:
                continue
            
            # Calculate accelerations between consecutive frames
            for i in range(1, len(path) - 1):
                # Velocity at frame i
                v1 = (path[i] - path[i-1]) / self.frame_time_s
                v2 = (path[i+1] - path[i]) / self.frame_time_s
                
                # Acceleration between frames
                accel = (v2 - v1) / self.frame_time_s
                
                # Check lateral acceleration (east-west component)
                lateral_accel = abs(accel[0])
                max_accel_found = max(max_accel_found, lateral_accel)
        
        # Allow small numerical tolerance for discrete sampling
        tolerance = 0.5  # m/s² - accounts for discrete sampling effects
        constraints_satisfied = max_accel_found <= (self.config.max_lateral_accel_mps2 + tolerance)
        
        return constraints_satisfied, max_accel_found
    
    def get_path_statistics(self) -> dict:
        """
        Get statistics about the generated paths for validation.
        
        Returns:
            Dictionary containing path statistics
        """
        if not self.paths:
            return {}
        
        stats = {
            'num_drones': len(self.paths),
            'path_length_frames': len(self.paths[0]),
            'duration_s': self.config.duration_s,
            'fps': self.config.fps
        }
        
        # Calculate motion statistics for first drone
        if self.paths:
            path = self.paths[0]
            east_positions = [pos[0] for pos in path]
            
            stats.update({
                'east_range_m': max(east_positions) - min(east_positions),
                'start_east_m': east_positions[0],
                'end_east_m': east_positions[-1],
                'max_east_m': max(east_positions),
                'min_east_m': min(east_positions)
            })
        
        # Verify acceleration constraints
        constraints_ok, max_accel = self.verify_acceleration_constraints()
        stats.update({
            'acceleration_constraints_satisfied': constraints_ok,
            'max_lateral_acceleration_mps2': max_accel,
            'configured_max_acceleration_mps2': self.config.max_lateral_accel_mps2
        })
        
        return stats