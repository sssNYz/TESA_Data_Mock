"""
Unit tests for motion generation functionality.

Tests smooth drone motion generation, physics-based constraints,
and acceleration limits.
"""

import pytest
import numpy as np
from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.motion import MotionGenerator


class TestMotionGenerator:
    """Test cases for MotionGenerator class."""
    
    def test_motion_generator_initialization(self):
        """Test MotionGenerator initializes correctly with default config."""
        config = SimulatorConfig()
        motion_gen = MotionGenerator(config)
        
        assert motion_gen.config == config
        assert len(motion_gen.paths) == config.num_drones
        assert motion_gen.total_frames == int(config.duration_s * config.fps)
        assert motion_gen.frame_time_s == 1.0 / config.fps
    
    def test_single_drone_path_generation(self):
        """Test path generation for a single drone."""
        config = SimulatorConfig(
            num_drones=1,
            duration_s=10.0,
            fps=10.0,
            path_span_m=20.0,
            speed_mps=2.0,
            max_lateral_accel_mps2=1.0
        )
        motion_gen = MotionGenerator(config)
        
        # Check path structure
        assert len(motion_gen.paths) == 1
        path = motion_gen.paths[0]
        expected_frames = int(config.duration_s * config.fps)
        assert len(path) == expected_frames
        
        # Check all positions are numpy arrays with 3 components (ENU)
        for pos in path:
            assert isinstance(pos, np.ndarray)
            assert pos.shape == (3,)
    
    def test_multiple_drone_paths(self):
        """Test path generation for multiple drones."""
        config = SimulatorConfig(
            num_drones=3,
            duration_s=8.0,
            fps=15.0
        )
        motion_gen = MotionGenerator(config)
        
        assert len(motion_gen.paths) == 3
        
        # All paths should have same length
        path_lengths = [len(path) for path in motion_gen.paths]
        assert all(length == path_lengths[0] for length in path_lengths)
        
        # Drones should have different north positions (staggered)
        first_positions = [path[0] for path in motion_gen.paths]
        north_positions = [pos[1] for pos in first_positions]
        
        # Should have different north offsets
        assert len(set(north_positions)) == len(north_positions)
    
    def test_left_to_right_motion(self):
        """Test that drones move from left to right (west to east)."""
        config = SimulatorConfig(
            num_drones=1,
            duration_s=10.0,
            fps=20.0,
            path_span_m=30.0,
            speed_mps=3.0
        )
        motion_gen = MotionGenerator(config)
        
        path = motion_gen.paths[0]
        east_positions = [pos[0] for pos in path]
        
        # Should start from negative east (left side)
        assert east_positions[0] < 0
        
        # Should generally increase (move east)
        # Allow for some initial acceleration phase
        mid_point = len(east_positions) // 2
        assert east_positions[mid_point] > east_positions[0]
        assert east_positions[-1] > east_positions[mid_point]
    
    def test_altitude_consistency(self):
        """Test that altitude remains consistent during flight."""
        config = SimulatorConfig(
            num_drones=2,
            path_altitude_agl_m=10.0
        )
        motion_gen = MotionGenerator(config)
        
        for drone_idx, path in enumerate(motion_gen.paths):
            altitudes = [pos[2] for pos in path]
            
            # Altitude should be consistent for each drone
            expected_altitude = config.path_altitude_agl_m + drone_idx * 0.5
            for altitude in altitudes:
                assert abs(altitude - expected_altitude) < 0.01
    
    def test_acceleration_constraints(self):
        """Test that acceleration constraints are respected."""
        config = SimulatorConfig(
            num_drones=1,
            duration_s=8.0,
            fps=30.0,
            speed_mps=5.0,
            max_lateral_accel_mps2=2.0,
            path_span_m=25.0
        )
        motion_gen = MotionGenerator(config)
        
        # Use built-in verification method
        constraints_ok, max_accel = motion_gen.verify_acceleration_constraints()
        
        assert constraints_ok, f"Acceleration constraint violated: {max_accel} > {config.max_lateral_accel_mps2}"
        assert max_accel <= config.max_lateral_accel_mps2 + 0.5  # Tolerance for discrete sampling
    
    def test_smooth_motion_no_teleporting(self):
        """Test that motion is smooth without sudden jumps."""
        config = SimulatorConfig(
            num_drones=1,
            duration_s=6.0,
            fps=25.0,
            speed_mps=4.0,
            max_lateral_accel_mps2=1.5
        )
        motion_gen = MotionGenerator(config)
        
        path = motion_gen.paths[0]
        frame_time = 1.0 / config.fps
        max_reasonable_speed = config.speed_mps * 1.7  # Allow some margin for discrete sampling
        
        # Check that consecutive positions don't have unrealistic jumps
        for i in range(1, len(path)):
            displacement = np.linalg.norm(path[i] - path[i-1])
            max_displacement = max_reasonable_speed * frame_time
            
            # Allow small tolerance for numerical precision
            tolerance = 0.02  # 2cm tolerance for discrete sampling effects
            assert displacement <= max_displacement + tolerance, \
                f"Unrealistic jump at frame {i}: {displacement} > {max_displacement + tolerance}"
    
    def test_get_positions_at_time(self):
        """Test getting positions at specific times."""
        config = SimulatorConfig(
            num_drones=2,
            duration_s=5.0,
            fps=10.0
        )
        motion_gen = MotionGenerator(config)
        
        # Test at start
        positions = motion_gen.get_positions_at_time(0.0)
        assert len(positions) == 2
        assert all(isinstance(pos, np.ndarray) for pos in positions)
        
        # Test at middle
        positions_mid = motion_gen.get_positions_at_time(2.5)
        assert len(positions_mid) == 2
        
        # Test at end
        positions_end = motion_gen.get_positions_at_time(5.0)
        assert len(positions_end) == 2
        
        # Positions should be different at different times
        assert not np.array_equal(positions[0], positions_mid[0])
    
    def test_get_positions_at_frame(self):
        """Test getting positions at specific frame indices."""
        config = SimulatorConfig(
            num_drones=1,
            duration_s=4.0,
            fps=20.0
        )
        motion_gen = MotionGenerator(config)
        
        total_frames = int(config.duration_s * config.fps)
        
        # Test first frame
        positions = motion_gen.get_positions_at_frame(0)
        assert len(positions) == 1
        
        # Test middle frame
        mid_frame = total_frames // 2
        positions_mid = motion_gen.get_positions_at_frame(mid_frame)
        assert len(positions_mid) == 1
        
        # Test last frame
        positions_last = motion_gen.get_positions_at_frame(total_frames - 1)
        assert len(positions_last) == 1
        
        # Test out of bounds (should return last position)
        positions_oob = motion_gen.get_positions_at_frame(total_frames + 10)
        assert np.array_equal(positions_oob[0], positions_last[0])
    
    def test_path_statistics(self):
        """Test path statistics generation."""
        config = SimulatorConfig(
            num_drones=2,
            duration_s=6.0,
            fps=15.0,
            path_span_m=20.0
        )
        motion_gen = MotionGenerator(config)
        
        stats = motion_gen.get_path_statistics()
        
        # Check required statistics
        assert stats['num_drones'] == 2
        assert stats['path_length_frames'] == int(config.duration_s * config.fps)
        assert stats['duration_s'] == config.duration_s
        assert stats['fps'] == config.fps
        
        # Check motion statistics
        assert 'east_range_m' in stats
        assert 'start_east_m' in stats
        assert 'end_east_m' in stats
        assert 'acceleration_constraints_satisfied' in stats
        assert 'max_lateral_acceleration_mps2' in stats
        
        # East range should be reasonable
        assert stats['east_range_m'] > 0
        assert stats['east_range_m'] <= config.path_span_m * 1.1  # Allow small margin
    
    def test_trapezoidal_velocity_profile(self):
        """Test that velocity follows trapezoidal profile for long paths."""
        config = SimulatorConfig(
            num_drones=1,
            duration_s=15.0,  # Long duration
            fps=50.0,  # High fps for precision
            path_span_m=50.0,  # Long path
            speed_mps=4.0,
            max_lateral_accel_mps2=2.0
        )
        motion_gen = MotionGenerator(config)
        
        path = motion_gen.paths[0]
        east_positions = [pos[0] for pos in path]
        frame_time = 1.0 / config.fps
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(east_positions)):
            velocity = (east_positions[i] - east_positions[i-1]) / frame_time
            velocities.append(velocity)
        
        # Should have acceleration phase, constant phase, deceleration phase
        max_velocity = max(velocities)
        
        # Max velocity should be close to configured speed
        assert abs(max_velocity - config.speed_mps) < 0.5
        
        # Should have periods of approximately constant velocity
        constant_velocity_count = sum(1 for v in velocities 
                                    if abs(v - max_velocity) < 0.2)
        assert constant_velocity_count > len(velocities) * 0.3  # At least 30% constant
    
    def test_triangular_velocity_profile_short_path(self):
        """Test triangular velocity profile for short paths that can't reach max speed."""
        config = SimulatorConfig(
            num_drones=1,
            duration_s=3.0,  # Short duration
            fps=30.0,
            path_span_m=5.0,  # Short path
            speed_mps=10.0,  # High speed (can't be reached)
            max_lateral_accel_mps2=2.0
        )
        motion_gen = MotionGenerator(config)
        
        path = motion_gen.paths[0]
        east_positions = [pos[0] for pos in path]
        frame_time = 1.0 / config.fps
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(east_positions)):
            velocity = (east_positions[i] - east_positions[i-1]) / frame_time
            velocities.append(velocity)
        
        # Max velocity should be less than configured speed
        max_velocity = max(velocities)
        assert max_velocity < config.speed_mps
        
        # Should have triangular profile (no constant velocity phase)
        # Peak should be in the middle portion
        peak_index = velocities.index(max_velocity)
        assert 0.3 * len(velocities) < peak_index < 0.7 * len(velocities)
    
    def test_deterministic_behavior(self):
        """Test that motion generation is deterministic."""
        config = SimulatorConfig(
            num_drones=2,
            deterministic_seed=42
        )
        
        # Generate paths twice with same config
        motion_gen1 = MotionGenerator(config)
        motion_gen2 = MotionGenerator(config)
        
        # Paths should be identical
        assert len(motion_gen1.paths) == len(motion_gen2.paths)
        
        for path1, path2 in zip(motion_gen1.paths, motion_gen2.paths):
            assert len(path1) == len(path2)
            for pos1, pos2 in zip(path1, path2):
                assert np.allclose(pos1, pos2, atol=1e-10)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very short duration
        config_short = SimulatorConfig(duration_s=0.1, fps=10.0)
        motion_gen_short = MotionGenerator(config_short)
        assert len(motion_gen_short.paths[0]) >= 1
        
        # Very low fps
        config_low_fps = SimulatorConfig(duration_s=2.0, fps=1.0)
        motion_gen_low_fps = MotionGenerator(config_low_fps)
        assert len(motion_gen_low_fps.paths[0]) == 2
        
        # Single frame
        config_single = SimulatorConfig(duration_s=1.0, fps=1.0)
        motion_gen_single = MotionGenerator(config_single)
        assert len(motion_gen_single.paths[0]) == 1
    
    def test_negative_time_handling(self):
        """Test handling of negative time values (before motion starts)."""
        config = SimulatorConfig(num_drones=1)
        motion_gen = MotionGenerator(config)
        
        # Test negative time
        positions = motion_gen.get_positions_at_time(-1.0)
        assert len(positions) == 1
        
        # Should return valid position (start position)
        start_positions = motion_gen.get_positions_at_time(0.0)
        assert np.allclose(positions[0], start_positions[0], atol=1e-6)


class TestMultiDroneMotionGeneration:
    """Test cases specifically for multi-drone motion generation (Requirement 7.1)."""
    
    def test_independent_drone_paths_generation(self):
        """Test that multiple drones generate independent smooth flight paths."""
        config = SimulatorConfig(
            num_drones=4,
            duration_s=8.0,
            fps=20.0,
            path_span_m=30.0,
            speed_mps=4.0
        )
        motion_gen = MotionGenerator(config)
        
        # Should generate one path per drone
        assert len(motion_gen.paths) == 4
        
        # All paths should have same length
        path_lengths = [len(path) for path in motion_gen.paths]
        assert all(length == path_lengths[0] for length in path_lengths)
        
        # Paths should be independent - different north positions
        initial_positions = [path[0] for path in motion_gen.paths]
        north_positions = [pos[1] for pos in initial_positions]
        
        # All north positions should be different
        assert len(set(north_positions)) == len(north_positions)
        
        # North positions should be spread out
        north_spread = max(north_positions) - min(north_positions)
        assert north_spread > 2.0  # At least 2m spread
    
    def test_drone_altitude_independence(self):
        """Test that drones have independent altitudes."""
        config = SimulatorConfig(
            num_drones=5,
            path_altitude_agl_m=10.0
        )
        motion_gen = MotionGenerator(config)
        
        # Get initial positions
        initial_positions = [path[0] for path in motion_gen.paths]
        altitudes = [pos[2] for pos in initial_positions]
        
        # Each drone should have different altitude
        assert len(set(altitudes)) == len(altitudes)
        
        # Altitudes should increase with drone index
        for i in range(1, len(altitudes)):
            assert altitudes[i] > altitudes[i-1]
        
        # Base altitude should match config
        assert abs(altitudes[0] - config.path_altitude_agl_m) < 0.01
    
    def test_staggered_start_times(self):
        """Test that drones have staggered start times for realistic motion."""
        config = SimulatorConfig(
            num_drones=3,
            duration_s=10.0,
            fps=20.0,
            path_span_m=20.0
        )
        motion_gen = MotionGenerator(config)
        
        # Check positions at early time - some drones should still be at start
        early_positions = motion_gen.get_positions_at_time(1.0)
        east_positions = [pos[0] for pos in early_positions]
        
        # Due to staggered starts, drones should be at different east positions
        assert len(set(east_positions)) >= 2  # At least some variation
        
        # First drone should be furthest along
        assert east_positions[0] >= east_positions[1]
        assert east_positions[1] >= east_positions[2]
    
    def test_multi_drone_acceleration_constraints(self):
        """Test that all drone paths respect acceleration constraints."""
        config = SimulatorConfig(
            num_drones=3,
            duration_s=6.0,
            fps=30.0,
            speed_mps=5.0,
            max_lateral_accel_mps2=2.0
        )
        motion_gen = MotionGenerator(config)
        
        # Check acceleration constraints for all drones
        constraints_ok, max_accel = motion_gen.verify_acceleration_constraints()
        
        assert constraints_ok, f"Multi-drone acceleration constraint violated: {max_accel} > {config.max_lateral_accel_mps2}"
        
        # Verify each path individually
        frame_time = 1.0 / config.fps
        for drone_idx, path in enumerate(motion_gen.paths):
            max_drone_accel = 0.0
            
            for i in range(1, len(path) - 1):
                v1 = (path[i] - path[i-1]) / frame_time
                v2 = (path[i+1] - path[i]) / frame_time
                accel = (v2 - v1) / frame_time
                lateral_accel = abs(accel[0])
                max_drone_accel = max(max_drone_accel, lateral_accel)
            
            assert max_drone_accel <= config.max_lateral_accel_mps2 + 0.5, \
                f"Drone {drone_idx} acceleration constraint violated: {max_drone_accel}"
    
    def test_multi_drone_path_independence_over_time(self):
        """Test that drone paths remain independent throughout simulation."""
        config = SimulatorConfig(
            num_drones=4,
            duration_s=8.0,
            fps=15.0
        )
        motion_gen = MotionGenerator(config)
        
        # Sample positions at different times
        test_times = [0.0, 2.0, 4.0, 6.0, 8.0]
        
        for time_s in test_times:
            positions = motion_gen.get_positions_at_time(time_s)
            
            # All drones should have different positions
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    distance = np.linalg.norm(positions[i] - positions[j])
                    assert distance > 0.1, f"Drones {i} and {j} too close at t={time_s}: {distance}m"
            
            # North positions should remain different (no crossing)
            north_positions = [pos[1] for pos in positions]
            assert len(set(north_positions)) == len(north_positions), \
                f"Drone north positions not unique at t={time_s}"
    
    def test_multi_drone_statistics(self):
        """Test path statistics for multi-drone scenarios."""
        config = SimulatorConfig(
            num_drones=3,
            duration_s=5.0,
            fps=20.0,
            path_span_m=25.0
        )
        motion_gen = MotionGenerator(config)
        
        stats = motion_gen.get_path_statistics()
        
        # Should report correct number of drones
        assert stats['num_drones'] == 3
        
        # Should have reasonable motion statistics
        assert stats['east_range_m'] > 0
        assert stats['east_range_m'] <= config.path_span_m * 1.1
        
        # Acceleration constraints should be satisfied
        assert stats['acceleration_constraints_satisfied'] is True
    
    def test_large_number_of_drones(self):
        """Test system with large number of drones."""
        config = SimulatorConfig(
            num_drones=10,
            duration_s=4.0,
            fps=10.0
        )
        motion_gen = MotionGenerator(config)
        
        # Should handle large number of drones
        assert len(motion_gen.paths) == 10
        
        # All paths should be valid
        for i, path in enumerate(motion_gen.paths):
            assert len(path) > 0
            assert all(isinstance(pos, np.ndarray) for pos in path)
            assert all(pos.shape == (3,) for pos in path)
        
        # Positions should be spread out
        positions = motion_gen.get_positions_at_time(2.0)
        north_positions = [pos[1] for pos in positions]
        north_spread = max(north_positions) - min(north_positions)
        assert north_spread > 5.0  # Should be well spread for 10 drones
    
    def test_single_vs_multi_drone_consistency(self):
        """Test that single drone behavior is consistent in multi-drone setup."""
        # Single drone config
        config_single = SimulatorConfig(num_drones=1, deterministic_seed=42)
        motion_gen_single = MotionGenerator(config_single)
        
        # Multi-drone config with same seed
        config_multi = SimulatorConfig(num_drones=3, deterministic_seed=42)
        motion_gen_multi = MotionGenerator(config_multi)
        
        # First drone in multi-drone should behave similarly to single drone
        # (allowing for slight differences due to staggered starts)
        single_path = motion_gen_single.paths[0]
        multi_first_path = motion_gen_multi.paths[0]
        
        assert len(single_path) == len(multi_first_path)
        
        # Check that the motion patterns are similar (same east progression)
        single_east = [pos[0] for pos in single_path]
        multi_east = [pos[0] for pos in multi_first_path]
        
        # Should have similar east progression (allowing for timing differences)
        single_range = max(single_east) - min(single_east)
        multi_range = max(multi_east) - min(multi_east)
        assert abs(single_range - multi_range) < 2.0  # Similar motion range


if __name__ == "__main__":
    pytest.main([__file__])