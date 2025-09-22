"""
Unit tests for the NoiseModel class.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.noise import NoiseModel


class TestNoiseModel:
    """Test cases for the NoiseModel class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            pixel_centroid_sigma_px=2.0,
            bbox_size_sigma_px=3.0,
            confidence_noise=0.1,
            miss_rate_small=0.05,
            false_positive_rate=0.02,
            deterministic_seed=42
        )
    
    @pytest.fixture
    def rng(self):
        """Create a deterministic random number generator."""
        return np.random.default_rng(42)
    
    @pytest.fixture
    def noise_model(self, config, rng):
        """Create a NoiseModel instance."""
        return NoiseModel(config, rng)
    
    def test_init(self, config, rng):
        """Test NoiseModel initialization."""
        noise_model = NoiseModel(config, rng)
        assert noise_model.config == config
        assert noise_model.rng == rng
    
    def test_apply_detection_noise_center_coordinates(self, noise_model):
        """Test that noise is applied to center coordinates."""
        clean_detection = {
            'center_px': [960.0, 540.0],
            'size_px': [40.0, 60.0],
            'bbox_px': [940.0, 510.0, 980.0, 570.0],
            'confidence': 0.9,
            'class': 'drone'
        }
        
        noisy_detection = noise_model.apply_detection_noise(clean_detection)
        
        # Center should be different due to noise
        assert noisy_detection['center_px'] != clean_detection['center_px']
        
        # But should still be within reasonable bounds
        center_x, center_y = noisy_detection['center_px']
        assert 0 <= center_x < 1920
        assert 0 <= center_y < 1080
        
        # Noise should be bounded (within ~3 sigma)
        original_x, original_y = clean_detection['center_px']
        assert abs(center_x - original_x) < 6.0  # 3 * sigma = 3 * 2.0
        assert abs(center_y - original_y) < 6.0
    
    def test_apply_detection_noise_bounding_box_size(self, noise_model):
        """Test that noise is applied to bounding box size."""
        clean_detection = {
            'center_px': [960.0, 540.0],
            'size_px': [40.0, 60.0],
            'bbox_px': [940.0, 510.0, 980.0, 570.0],
            'confidence': 0.9,
            'class': 'drone'
        }
        
        noisy_detection = noise_model.apply_detection_noise(clean_detection)
        
        # Size should be different due to noise
        assert noisy_detection['size_px'] != clean_detection['size_px']
        
        # Size should remain positive
        width, height = noisy_detection['size_px']
        assert width >= 1.0
        assert height >= 1.0
        
        # Bounding box should be updated based on noisy center and size
        assert 'bbox_px' in noisy_detection
        x1, y1, x2, y2 = noisy_detection['bbox_px']
        assert x1 < x2
        assert y1 < y2
        assert 0 <= x1 < 1920
        assert 0 <= y1 < 1080
        assert 0 < x2 <= 1920
        assert 0 < y2 <= 1080
    
    def test_apply_detection_noise_confidence(self, noise_model):
        """Test that noise is applied to confidence scores."""
        clean_detection = {
            'center_px': [960.0, 540.0],
            'size_px': [40.0, 60.0],
            'confidence': 0.8,
            'class': 'drone'
        }
        
        noisy_detection = noise_model.apply_detection_noise(clean_detection)
        
        # Confidence should be different due to noise
        assert noisy_detection['confidence'] != clean_detection['confidence']
        
        # Confidence should remain in valid range [0, 1]
        assert 0.0 <= noisy_detection['confidence'] <= 1.0
    
    def test_apply_detection_noise_preserves_class(self, noise_model):
        """Test that class information is preserved."""
        clean_detection = {
            'center_px': [960.0, 540.0],
            'size_px': [40.0, 60.0],
            'confidence': 0.8,
            'class': 'drone'
        }
        
        noisy_detection = noise_model.apply_detection_noise(clean_detection)
        assert noisy_detection['class'] == 'drone'
    
    def test_apply_detection_noise_statistical_properties(self, config):
        """Test statistical properties of applied noise."""
        rng = np.random.default_rng(42)
        noise_model = NoiseModel(config, rng)
        
        # Generate many noisy detections to test statistical properties
        clean_detection = {
            'center_px': [960.0, 540.0],
            'size_px': [40.0, 60.0],
            'confidence': 0.8,
            'class': 'drone'
        }
        
        num_samples = 1000
        center_x_values = []
        center_y_values = []
        width_values = []
        height_values = []
        confidence_values = []
        
        for _ in range(num_samples):
            noisy = noise_model.apply_detection_noise(clean_detection)
            center_x_values.append(noisy['center_px'][0])
            center_y_values.append(noisy['center_px'][1])
            width_values.append(noisy['size_px'][0])
            height_values.append(noisy['size_px'][1])
            confidence_values.append(noisy['confidence'])
        
        # Test that noise is approximately centered around original values
        assert abs(np.mean(center_x_values) - 960.0) < 0.2
        assert abs(np.mean(center_y_values) - 540.0) < 0.2
        assert abs(np.mean(width_values) - 40.0) < 0.2
        assert abs(np.mean(height_values) - 60.0) < 0.2
        assert abs(np.mean(confidence_values) - 0.8) < 0.02
        
        # Test that standard deviations are approximately correct
        assert abs(np.std(center_x_values) - 2.0) < 0.2
        assert abs(np.std(center_y_values) - 2.0) < 0.2
        assert abs(np.std(width_values) - 3.0) < 0.3
        assert abs(np.std(height_values) - 3.0) < 0.3
        assert abs(np.std(confidence_values) - 0.1) < 0.02
    
    def test_should_miss_detection_small_objects(self, noise_model):
        """Test miss detection probability for small objects."""
        # Small object (10x10 = 100 pixels²)
        small_detection = {
            'size_px': [10.0, 10.0],
            'class': 'drone'
        }
        
        # Test many times to verify statistical behavior
        num_tests = 1000
        misses = 0
        
        for _ in range(num_tests):
            if noise_model.should_miss_detection(small_detection):
                misses += 1
        
        miss_rate = misses / num_tests
        
        # Should have higher miss rate for small objects
        # Expected rate should be around miss_rate_small (0.05) but could be higher due to size scaling
        assert 0.02 < miss_rate < 0.15  # Allow some variance
    
    def test_should_miss_detection_large_objects(self, noise_model):
        """Test miss detection probability for large objects."""
        # Large object (50x50 = 2500 pixels²)
        large_detection = {
            'size_px': [50.0, 50.0],
            'class': 'drone'
        }
        
        # Test many times to verify statistical behavior
        num_tests = 1000
        misses = 0
        
        for _ in range(num_tests):
            if noise_model.should_miss_detection(large_detection):
                misses += 1
        
        miss_rate = misses / num_tests
        
        # Should have lower miss rate for large objects
        # Expected rate should be around miss_rate_small * 0.1 = 0.005
        assert miss_rate < 0.02
    
    def test_should_miss_detection_very_small_objects(self, noise_model):
        """Test miss detection probability for very small objects."""
        # Very small object (5x5 = 25 pixels²)
        very_small_detection = {
            'size_px': [5.0, 5.0],
            'class': 'drone'
        }
        
        # Test many times to verify statistical behavior
        num_tests = 1000
        misses = 0
        
        for _ in range(num_tests):
            if noise_model.should_miss_detection(very_small_detection):
                misses += 1
        
        miss_rate = misses / num_tests
        
        # Should have even higher miss rate for very small objects
        # Expected rate should be around miss_rate_small * 2 = 0.1
        assert 0.05 < miss_rate < 0.25  # Allow some variance
    
    def test_should_miss_detection_no_size_info(self, noise_model):
        """Test miss detection when size information is missing."""
        detection_no_size = {
            'class': 'drone',
            'confidence': 0.8
        }
        
        # Should not miss detection when size info is missing
        assert not noise_model.should_miss_detection(detection_no_size)
    
    def test_generate_false_positive_probability(self, noise_model):
        """Test false positive generation probability."""
        num_tests = 1000
        false_positives = 0
        
        for _ in range(num_tests):
            fp = noise_model.generate_false_positive()
            if fp is not None:
                false_positives += 1
        
        fp_rate = false_positives / num_tests
        
        # Should match configured false positive rate (0.02)
        assert 0.01 < fp_rate < 0.04  # Allow some variance
    
    def test_generate_false_positive_properties(self, noise_model):
        """Test properties of generated false positives."""
        # Generate multiple false positives to test properties
        false_positives = []
        
        # Force generation by temporarily increasing rate
        original_rate = noise_model.config.false_positive_rate
        noise_model.config.false_positive_rate = 1.0  # Always generate
        
        try:
            for _ in range(100):
                fp = noise_model.generate_false_positive()
                if fp is not None:
                    false_positives.append(fp)
        finally:
            noise_model.config.false_positive_rate = original_rate
        
        assert len(false_positives) > 50  # Should generate many with rate = 1.0
        
        for fp in false_positives[:10]:  # Test first 10
            # Should have correct class
            assert fp['class'] == 'false_drone'
            
            # Should have low confidence
            assert 0.1 <= fp['confidence'] <= 0.4
            
            # Should have reasonable size
            width, height = fp['size_px']
            assert 5.0 <= width <= 30.0
            assert 5.0 <= height <= 30.0
            
            # Should be within image bounds
            center_x, center_y = fp['center_px']
            assert 0 <= center_x < 1920
            assert 0 <= center_y < 1080
            
            # Bounding box should be valid
            x1, y1, x2, y2 = fp['bbox_px']
            assert x1 < x2
            assert y1 < y2
            assert 0 <= x1 < 1920
            assert 0 <= y1 < 1080
            assert 0 < x2 <= 1920
            assert 0 < y2 <= 1080
    
    def test_apply_confidence_variation_large_objects(self, noise_model):
        """Test confidence variation for large objects."""
        base_confidence = 0.7
        large_size = (40.0, 40.0)  # 1600 pixels² > 1000 threshold
        
        # Test multiple times to see the effect
        confidences = []
        for _ in range(100):
            conf = noise_model.apply_confidence_variation(base_confidence, large_size)
            confidences.append(conf)
        
        mean_confidence = np.mean(confidences)
        
        # Large objects should get confidence boost
        assert mean_confidence > base_confidence
        assert all(0.0 <= c <= 1.0 for c in confidences)
    
    def test_apply_confidence_variation_small_objects(self, noise_model):
        """Test confidence variation for small objects."""
        base_confidence = 0.7
        small_size = (8.0, 8.0)  # 64 pixels² < 100 threshold
        
        # Test multiple times to see the effect
        confidences = []
        for _ in range(100):
            conf = noise_model.apply_confidence_variation(base_confidence, small_size)
            confidences.append(conf)
        
        mean_confidence = np.mean(confidences)
        
        # Small objects should get confidence penalty
        assert mean_confidence < base_confidence
        assert all(0.0 <= c <= 1.0 for c in confidences)
    
    def test_apply_confidence_variation_medium_objects(self, noise_model):
        """Test confidence variation for medium-sized objects."""
        base_confidence = 0.7
        medium_size = (20.0, 20.0)  # 400 pixels² (between thresholds)
        
        # Test multiple times to see the effect
        confidences = []
        for _ in range(100):
            conf = noise_model.apply_confidence_variation(base_confidence, medium_size)
            confidences.append(conf)
        
        mean_confidence = np.mean(confidences)
        
        # Medium objects should keep approximately base confidence
        assert abs(mean_confidence - base_confidence) < 0.05
        assert all(0.0 <= c <= 1.0 for c in confidences)
    
    def test_get_noise_statistics(self, noise_model, config):
        """Test noise statistics reporting."""
        stats = noise_model.get_noise_statistics()
        
        # Should contain all expected parameters
        expected_keys = [
            'pixel_centroid_sigma_px',
            'bbox_size_sigma_px',
            'confidence_noise',
            'miss_rate_small',
            'false_positive_rate',
            'image_dimensions'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        # Values should match configuration
        assert stats['pixel_centroid_sigma_px'] == config.pixel_centroid_sigma_px
        assert stats['bbox_size_sigma_px'] == config.bbox_size_sigma_px
        assert stats['confidence_noise'] == config.confidence_noise
        assert stats['miss_rate_small'] == config.miss_rate_small
        assert stats['false_positive_rate'] == config.false_positive_rate
        assert stats['image_dimensions'] == [config.image_width_px, config.image_height_px]
    
    def test_deterministic_behavior(self, config):
        """Test that noise model produces deterministic results with fixed seed."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        noise_model1 = NoiseModel(config, rng1)
        noise_model2 = NoiseModel(config, rng2)
        
        clean_detection = {
            'center_px': [960.0, 540.0],
            'size_px': [40.0, 60.0],
            'confidence': 0.8,
            'class': 'drone'
        }
        
        # Should produce identical results with same seed
        noisy1 = noise_model1.apply_detection_noise(clean_detection)
        noisy2 = noise_model2.apply_detection_noise(clean_detection)
        
        assert noisy1['center_px'] == noisy2['center_px']
        assert noisy1['size_px'] == noisy2['size_px']
        assert noisy1['confidence'] == noisy2['confidence']
    
    def test_boundary_conditions(self, noise_model):
        """Test noise application at image boundaries."""
        # Detection near image edge
        edge_detection = {
            'center_px': [10.0, 10.0],  # Near top-left corner
            'size_px': [20.0, 20.0],
            'confidence': 0.8,
            'class': 'drone'
        }
        
        noisy_detection = noise_model.apply_detection_noise(edge_detection)
        
        # Should remain within image bounds
        center_x, center_y = noisy_detection['center_px']
        assert 0 <= center_x < 1920
        assert 0 <= center_y < 1080
        
        # Bounding box should also be within bounds
        x1, y1, x2, y2 = noisy_detection['bbox_px']
        assert 0 <= x1 < 1920
        assert 0 <= y1 < 1080
        assert 0 < x2 <= 1920
        assert 0 < y2 <= 1080