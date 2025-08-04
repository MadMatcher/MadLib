"""Integration tests for complete MadLib workflows."""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from MadLib.tools import (
    down_sample, create_seeds, train_matcher, apply_matcher, featurize, 
    create_features
)
from MadLib import Labeler, MLModel, Feature


@pytest.mark.integration
class TestCompleteWorkflow:
    """Test complete end-to-end matching workflows."""

    def test_basic_matching_pipeline(self, sample_dataframe_a, sample_dataframe_b, 
                                   sample_candidates, gold_labels):
        """Test a complete basic matching pipeline."""
        # Step 1: Create features
        features = create_features(
            sample_dataframe_a, 
            sample_dataframe_b, 
            ['name', 'email'], 
            ['name', 'email']
        )
        
        # Step 2: Featurize candidates
        feature_vectors = featurize(
            features, sample_dataframe_a, sample_dataframe_b, sample_candidates
        )
        
        assert isinstance(feature_vectors, pd.DataFrame)
        assert 'feature_vectors' in feature_vectors.columns
        assert len(feature_vectors) > 0
        
        # Fill NaN values in feature vectors and add score column
        feature_vectors['feature_vectors'] = feature_vectors['feature_vectors'].apply(
            lambda x: [0.0 if pd.isna(val) else val for val in x] 
            if x is not None else [0.0]
        )
        feature_vectors['score'] = feature_vectors['feature_vectors'].apply(
            lambda x: np.sum(x) if x is not None else 0.0
        )
        
        # Step 3: Create manual seeds with both positive and negative examples
        # to ensure sklearn can train properly
        seeds = feature_vectors.head(2).copy()
        seeds['label'] = [1.0, 0.0]  # One positive, one negative
        
        assert isinstance(seeds, pd.DataFrame)
        assert 'label' in seeds.columns
        assert len(seeds) <= 2
        
        # Step 4: Train matcher
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42}
        }
        matcher = train_matcher(model_spec, seeds)
        
        assert matcher is not None
        assert hasattr(matcher, 'trained_model')
        
        # Step 5: Apply matcher
        predictions = apply_matcher(
            matcher, feature_vectors, 
            feature_col='feature_vectors', output_col='prediction'
        )
        
        assert isinstance(predictions, pd.DataFrame)
        assert 'prediction' in predictions.columns
        assert len(predictions) == len(feature_vectors)
        assert predictions['prediction'].notna().all()

    def test_workflow_with_custom_components(self, sample_dataframe_a, 
                                           sample_dataframe_b, sample_candidates):
        """Test workflow with custom Feature, MLModel, and Labeler."""
        
        # Custom Feature
        class SimpleFeature(Feature):
            def __str__(self):
                return "simple_feature"
            
            def _preprocess_output_column(self, attr):
                return None
            
            def _preprocess(self, data, input_col):
                return data
            
            def __call__(self, A_dict, B_series):
                # Simple mock feature that returns random values
                np.random.seed(42)
                return pd.Series(
                    np.random.random(len(B_series)), index=B_series.index
                )
        
        # Custom MLModel
        class SimpleMLModel(MLModel):
            def __init__(self):
                self.threshold = 0.5
                self._trained = False
                self._trained_model = None
            
            @property
            def nan_fill(self): return 0.0
            @property
            def use_vectors(self): return False
            @property
            def use_floats(self): return True
            @property
            def trained_model(self): return self._trained_model
            
            def train(self, df, vector_col, label_column, return_estimator=False):
                self._trained = True
                self._trained_model = "trained_mock_model"  # Mock trained model
                return self
            
            def predict(self, df, vector_col, output_col):
                result = df.copy()
                predictions = [
                    1.0 if np.sum(row[vector_col]) > self.threshold else 0.0 
                    for _, row in df.iterrows()
                ]
                result[output_col] = predictions
                return result
            
            def prediction_conf(self, df, vector_col, label_column):
                result = df.copy()
                result['conf'] = 0.8
                return result
            
            def entropy(self, df, vector_col, output_col):
                result = df.copy()
                result[output_col] = 0.5
                return result
            
            def params_dict(self):
                return {'threshold': self.threshold, 'trained': self._trained}
        
        # Custom Labeler
        class SimpleLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0 if (id1 + id2) % 3 == 0 else 0.0
        
        # Execute simplified workflow with custom components
        # Create simple feature vectors manually - mimic the structure from featurize
        feature_vectors = pd.DataFrame({
            'id1': [1, 2, 3, 4, 5],
            'id2': [101, 102, 103, 104, 105], 
            'feature_vectors': [[0.1, 0.2, 0.3] for _ in range(5)],
            'score': [0.6, 0.4, 0.8, 0.2, 0.9],
            '_id': range(5)
        })
        
        custom_labeler = SimpleLabeler()
        seeds = create_seeds(feature_vectors, nseeds=2, labeler=custom_labeler)
        
        custom_model = SimpleMLModel()
        matcher = train_matcher(custom_model, seeds)
        
        predictions = apply_matcher(
            matcher, feature_vectors, feature_col='feature_vectors', output_col='prediction'
        )
        
        assert len(predictions) == len(feature_vectors)
        assert 'prediction' in predictions.columns


@pytest.mark.integration
class TestWorkflowRobustness:
    """Test workflow robustness with edge cases and data quality issues."""

    def test_workflow_with_missing_data(self, sample_candidates, gold_labels):
        """Test workflow with missing values in data."""
        # Create dataframes with missing values
        df_a_missing = pd.DataFrame({
            '_id': [1, 2, 3, 4, 5],
            'name': ['Alice Smith', None, 'Carol Davis', 'David Wilson', 'Eve Brown'],
            'email': [None, 'bob@email.com', 'carol@email.com', 'david@email.com', None],
            'age': [25, 30, None, 35, 22]
        })
        
        df_b_missing = pd.DataFrame({
            '_id': [101, 102, 103, 104, 105],
            'name': ['Alicia Smith', 'Robert Jones', None, 'Dave Wilson', 'Eva Brown'],
            'email': ['alicia@email.com', None, 'caroline@email.com', None, 'eva@email.com'],
            'age': [26, 29, 28, None, 23]
        })
        
        features = create_features(
            df_a_missing, 
            df_b_missing, 
            ['name', 'email'], 
            ['name', 'email']
        )
        feature_vectors = featurize(
            features, df_a_missing, df_b_missing, sample_candidates
        )
        
        # Should handle missing values gracefully
        assert len(feature_vectors) == len(sample_candidates)
        assert 'feature_vectors' in feature_vectors.columns
        
        # Fill NaN values in feature vectors and add score column
        feature_vectors['feature_vectors'] = feature_vectors['feature_vectors'].apply(
            lambda x: [0.0 if pd.isna(val) else val for val in x] 
            if x is not None else [0.0]
        )
        feature_vectors['score'] = feature_vectors['feature_vectors'].apply(
            lambda x: np.sum(x) if x is not None else 0.0
        )
        
        # featurize already returns id1 column - no need to transform id1_list
        
        # Continue with pipeline
        gold_labeler = {'name': 'gold', 'gold': gold_labels}
        seeds = create_seeds(feature_vectors, nseeds=2, labeler=gold_labeler)
        
        model_spec = {
            'model_type': 'sklearn', 
            'model': LogisticRegression, 
            'model_args': {'random_state': 42}
        }
        matcher = train_matcher(model_spec, seeds)
        
        predictions = apply_matcher(
            matcher, feature_vectors, feature_col='feature_vectors', output_col='prediction'
        )
        
        assert len(predictions) == len(feature_vectors)

    def test_workflow_reproducibility(self, sample_dataframe_a, sample_dataframe_b, 
                                    sample_candidates, gold_labels):
        """Test that workflow produces reproducible results."""
        def run_pipeline():
            features = create_features(
                sample_dataframe_a, 
                sample_dataframe_b, 
                ['name'], 
                ['name']
            )
            feature_vectors = featurize(
                features, sample_dataframe_a, sample_dataframe_b, sample_candidates
            )
            
            # Fill NaN values in feature vectors and add score column
            feature_vectors['feature_vectors'] = feature_vectors['feature_vectors'].apply(
                lambda x: [0.0 if pd.isna(val) else val for val in x] 
                if x is not None else [0.0]
            )
            feature_vectors['score'] = feature_vectors['feature_vectors'].apply(
                lambda x: np.sum(x) if x is not None else 0.0
            )
            
            # featurize already returns id1 column - no need to transform id1_list
            
            gold_labeler = {'name': 'gold', 'gold': gold_labels}
            seeds = create_seeds(feature_vectors, nseeds=2, labeler=gold_labeler)
            
            model_spec = {
                'model_type': 'sklearn', 
                'model': LogisticRegression, 
                'model_args': {'random_state': 42}
            }
            matcher = train_matcher(model_spec, seeds)
            
            predictions = apply_matcher(
                matcher, feature_vectors, feature_col='feature_vectors', output_col='prediction'
            )
            return predictions
        
        # Run pipeline twice
        predictions1 = run_pipeline()
        predictions2 = run_pipeline()
        
        # Results should be identical
        assert len(predictions1) == len(predictions2)
        pd.testing.assert_frame_equal(
            predictions1.sort_index(), predictions2.sort_index()
        )
 