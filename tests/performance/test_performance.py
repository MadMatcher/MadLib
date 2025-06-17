"""Performance tests for MadMatcher scalability and efficiency."""

import pytest
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression

from madmatcher_tools.tools import down_sample, create_seeds, train_matcher, apply_matcher


@pytest.mark.performance
class TestBasicPerformance:
    """Basic performance tests that work with current API."""

    @pytest.mark.parametrize("n_candidates", [50, 200, 500])
    def test_downsampling_performance(self, n_candidates, performance_timer):
        """Test down sampling performance with different candidate set sizes."""
        # Create large candidate set
        feature_vectors = pd.DataFrame({
            'id1': list(range(n_candidates)),
            'id2': list(range(1000, 1000 + n_candidates)),
            'features': [np.random.random(10).tolist() for _ in range(n_candidates)],
            'score': np.random.random(n_candidates)
        })
        
        performance_timer.start()
        sampled = down_sample(feature_vectors, percent=0.3, search_id_column='id2')
        elapsed = performance_timer.stop()
        
        # Performance assertions
        expected_size = int(0.3 * n_candidates)
        assert len(sampled) <= n_candidates
        assert len(sampled) >= expected_size - 10  # Allow some variance
        
        # Should be fast regardless of size
        assert elapsed < 5.0
        
        print(f"Downsampling {n_candidates} candidates took {elapsed:.3f} seconds")

    @pytest.mark.parametrize("n_features", [5, 20, 50])
    def test_model_training_performance(self, n_features, performance_timer):
        """Test model training performance with different feature dimensions."""
        # Create training data with varying feature dimensions
        n_samples = 100
        training_data = pd.DataFrame({
            'features': [np.random.random(n_features).tolist() for _ in range(n_samples)],
            'label': np.random.choice([0.0, 1.0], n_samples)
        })
        
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42, 'max_iter': 100}
        }
        
        performance_timer.start()
        matcher = train_matcher(model_spec, training_data)
        elapsed = performance_timer.stop()
        
        # Performance assertions
        assert matcher is not None
        assert hasattr(matcher, 'trained_model')
        
        # Training should be fast for reasonable feature dimensions
        assert elapsed < 10.0
        
        print(f"Training with {n_features} features took {elapsed:.3f} seconds")

    def test_feature_vector_efficiency(self):
        """Test efficiency of feature vector storage and manipulation."""
        # Create many feature vectors
        n_vectors = 1000
        feature_dim = 20
        
        start_time = time.time()
        
        feature_vectors = pd.DataFrame({
            'id1': range(n_vectors),
            'id2': range(1000, 1000 + n_vectors),
            'features': [np.random.random(feature_dim).tolist() for _ in range(n_vectors)]
        })
        
        # Test various operations
        # 1. Access features
        first_features = feature_vectors['features'].iloc[0]
        assert len(first_features) == feature_dim
        
        # 2. Filter vectors
        filtered = feature_vectors[feature_vectors['id1'] < 100]
        assert len(filtered) == 100
        
        # 3. Apply transformations
        feature_sums = feature_vectors['features'].apply(lambda x: sum(x))
        assert len(feature_sums) == n_vectors
        
        elapsed = time.time() - start_time
        
        # Operations should be fast
        assert elapsed < 5.0
        
        print(f"Feature vector operations took {elapsed:.3f} seconds") 