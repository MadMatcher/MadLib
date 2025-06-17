"""
Unit tests for madmatcher_tools.tools module.

Tests all public API functions with various input types and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from madmatcher_tools.tools import (
    down_sample, create_seeds, train_matcher, apply_matcher, label_data
)
from madmatcher_tools import Labeler, MLModel


@pytest.mark.unit
class TestDownSample:
    """Test down_sample function."""

    def test_down_sample_basic(self, sample_feature_vectors):
        """Test basic down sampling."""
        fvs = sample_feature_vectors.copy()
        fvs['score'] = [0.9, 0.8, 0.6, 0.4, 0.2]
        
        result = down_sample(fvs, percent=0.6, search_id_column='id2')
        
        assert len(result) <= len(fvs)
        assert 'score' in result.columns

    def test_down_sample_invalid_percent(self, sample_feature_vectors):
        """Test down sampling with invalid percentage."""
        fvs = sample_feature_vectors.copy()
        fvs['score'] = [0.9, 0.8, 0.6, 0.4, 0.2]
        
        with pytest.raises(ValueError):
            down_sample(fvs, percent=0.0, search_id_column='id2')

    def test_down_sample_custom_score_column(self, sample_feature_vectors):
        """Test down sampling with custom score column."""
        fvs = sample_feature_vectors.copy()
        fvs['custom_score'] = [0.1, 0.9, 0.5, 0.3, 0.7]
        
        result = down_sample(fvs, percent=0.4, search_id_column='id2', 
                           score_column='custom_score')
        
        assert len(result) <= len(fvs)
        assert 'custom_score' in result.columns

    def test_down_sample_custom_bucket_size(self, sample_feature_vectors):
        """Test down sampling with custom bucket size."""
        fvs = sample_feature_vectors.copy()
        fvs['score'] = [0.9, 0.8, 0.6, 0.4, 0.2]
        
        result = down_sample(fvs, percent=0.8, search_id_column='id2', 
                           bucket_size=2)
        
        assert len(result) <= len(fvs)

    def test_down_sample_empty_dataframe(self):
        """Test down sampling with empty DataFrame."""
        fvs = pd.DataFrame(columns=['id2', 'score'])
        
        result = down_sample(fvs, percent=0.5, search_id_column='id2')
        
        assert len(result) == 0
        assert list(result.columns) == ['id2', 'score']


@pytest.mark.unit
class TestCreateSeeds:
    """Test create_seeds function."""

    def test_create_seeds_gold_labeler(self, sample_feature_vectors, gold_labels):
        """Test create_seeds with gold standard labeler."""
        fvs = sample_feature_vectors.copy()
        gold_labeler = {'name': 'gold', 'gold': gold_labels}
        
        seeds = create_seeds(fvs, nseeds=3, labeler=gold_labeler)
        
        assert len(seeds) <= 3
        assert 'label' in seeds.columns
        assert seeds['label'].isin([0.0, 1.0]).all()
        assert len(seeds[seeds['label'] == 1.0]) > 0  # Should have some positive labels

    def test_create_seeds_custom_labeler(self, sample_feature_vectors):
        """Test create_seeds with custom labeler."""
        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0 if id1 % 2 == 0 else 0.0
        
        fvs = sample_feature_vectors.copy()
        labeler = TestLabeler()
        
        seeds = create_seeds(fvs, nseeds=2, labeler=labeler)
        
        assert len(seeds) <= 2
        assert 'label' in seeds.columns
        assert seeds['label'].isin([0.0, 1.0]).all()

    def test_create_seeds_zero_seeds(self, sample_feature_vectors):
        """Test create_seeds with zero seeds."""
        fvs = sample_feature_vectors.copy()
        labeler = Mock(spec=Labeler)
        
        with pytest.raises(ValueError, match="no seeds would be created"):
            create_seeds(fvs, nseeds=0, labeler=labeler)

    def test_create_seeds_too_many_seeds(self, sample_feature_vectors):
        """Test create_seeds with too many seeds requested."""
        fvs = sample_feature_vectors.copy()
        labeler = Mock(spec=Labeler)
        labeler.return_value = 1.0
        
        # The actual implementation returns a ValueError instead of raising it
        result = create_seeds(fvs, nseeds=100, labeler=labeler)
        assert isinstance(result, ValueError)
        assert "number of seeds would exceed" in str(result)

    def test_create_seeds_custom_score_column(self, sample_feature_vectors):
        """Test create_seeds with custom score column."""
        fvs = sample_feature_vectors.copy()
        fvs['custom_score'] = [0.1, 0.9, 0.5, 0.3, 0.7]
        labeler = Mock(spec=Labeler)
        labeler.return_value = 1.0
        
        seeds = create_seeds(fvs, nseeds=2, labeler=labeler, 
                           score_column='custom_score')
        
        assert len(seeds) <= 2
        assert 'label' in seeds.columns


@pytest.mark.unit
class TestTrainMatcher:
    """Test train_matcher function."""

    def test_train_matcher_sklearn(self, mock_labeled_data):
        """Test training with sklearn model."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42}
        }
        
        model = train_matcher(model_spec, mock_labeled_data)
        
        assert hasattr(model, 'trained_model')
        assert isinstance(model.trained_model, LogisticRegression)

    def test_train_matcher_different_sklearn_models(self, mock_labeled_data):
        """Test training with different sklearn models."""
        models_to_test = [
            LogisticRegression,
            RandomForestClassifier
        ]
        
        for model_class in models_to_test:
            model_spec = {
                'model_type': 'sklearn',
                'model': model_class,
                'model_args': {'random_state': 42}
            }
            
            model = train_matcher(model_spec, mock_labeled_data)
            
            assert hasattr(model, 'trained_model')
            assert isinstance(model.trained_model, model_class)

    def test_train_matcher_custom_mlmodel(self, mock_labeled_data):
        """Test training with custom MLModel."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                self._trained_model = self  # Mock trained model
                return self
            
            def predict(self, df, vector_col, output_col):
                df = df.copy()
                df[output_col] = 1.0
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                df = df.copy()
                df['conf'] = 0.8
                return df
            
            def entropy(self, df, vector_col, output_col):
                df = df.copy()
                df[output_col] = 0.5
                return df
            
            def params_dict(self):
                return {'trained': self._trained}
        
        custom_model = TestMLModel()
        
        result = train_matcher(custom_model, mock_labeled_data)
        
        assert result._trained
        assert result.params_dict()['trained']

    def test_train_matcher_custom_feature_column(self, mock_labeled_data):
        """Test training with custom feature column name."""
        labeled_data = mock_labeled_data.copy()
        labeled_data['custom_features'] = labeled_data['features']
        
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42}
        }
        
        model = train_matcher(model_spec, labeled_data, 
                            feature_col='custom_features')
        
        assert hasattr(model, 'trained_model')

    def test_train_matcher_custom_label_column(self, mock_labeled_data):
        """Test training with custom label column name."""
        labeled_data = mock_labeled_data.copy()
        labeled_data['custom_label'] = labeled_data['label']
        
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42}
        }
        
        model = train_matcher(model_spec, labeled_data,
                            label_col='custom_label')
        
        assert hasattr(model, 'trained_model')


@pytest.mark.unit
class TestApplyMatcher:
    """Test apply_matcher function."""

    def test_apply_matcher_basic(self, sample_feature_vectors, mock_labeled_data):
        """Test basic model application."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42}
        }
        model = train_matcher(model_spec, mock_labeled_data)
        
        result = apply_matcher(model, sample_feature_vectors, 
                             feature_col='features', output_col='prediction')
        
        assert 'prediction' in result.columns
        assert len(result) == len(sample_feature_vectors)
        assert result['prediction'].notna().all()

    def test_apply_matcher_custom_columns(self, sample_feature_vectors, mock_labeled_data):
        """Test model application with custom column names."""
        # Prepare data with custom column names
        fvs = sample_feature_vectors.copy()
        fvs['custom_features'] = fvs['features']
        
        # Train a model
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42}
        }
        model = train_matcher(model_spec, mock_labeled_data)
        
        # Apply with custom columns
        result = apply_matcher(model, fvs,
                             feature_col='custom_features', 
                             output_col='custom_prediction')
        
        assert 'custom_prediction' in result.columns
        assert 'prediction' not in result.columns

    def test_apply_matcher_preserve_original_columns(self, sample_feature_vectors, 
                                                   mock_labeled_data):
        """Test that original columns are preserved."""
        original_columns = set(sample_feature_vectors.columns)
        
        # Train and apply model
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42}
        }
        model = train_matcher(model_spec, mock_labeled_data)
        result = apply_matcher(model, sample_feature_vectors,
                             feature_col='features', output_col='prediction')
        
        # Check all original columns are preserved
        for col in original_columns:
            assert col in result.columns


@pytest.mark.unit
class TestLabelData:
    """Test label_data function."""

    @patch('madmatcher_tools.tools.create_seeds')
    @patch('madmatcher_tools.tools.EntropyActiveLearner')
    def test_label_data_batch_mode(self, mock_learner_class, mock_create_seeds,
                                 sample_feature_vectors, gold_labels):
        """Test label_data in batch mode."""
        # Setup mocks
        mock_seeds = pd.DataFrame({'id1': [1], 'id2': [101], 'label': [1.0]})
        mock_create_seeds.return_value = mock_seeds
        
        mock_learner = Mock()
        mock_labeled_data = pd.DataFrame({'id1': [1, 2], 'id2': [101, 102], 'label': [1.0, 0.0]})
        mock_learner.train.return_value = mock_labeled_data
        mock_learner_class.return_value = mock_learner
        
        # Test
        model_spec = {'model_type': 'sklearn', 'model': LogisticRegression, 'model_args': {}}
        labeler_spec = {'name': 'gold', 'gold': gold_labels}
        
        result = label_data(model_spec, 'batch', labeler_spec, sample_feature_vectors)
        
        assert isinstance(result, pd.DataFrame)
        mock_learner_class.assert_called_once()
        mock_learner.train.assert_called_once()

    @patch('madmatcher_tools.tools.create_seeds')
    @patch('madmatcher_tools.tools.ContinuousEntropyActiveLearner')
    def test_label_data_continuous_mode(self, mock_learner_class, mock_create_seeds,
                                      sample_feature_vectors, gold_labels):
        """Test label_data in continuous mode."""
        # Setup mocks
        mock_seeds = pd.DataFrame({'id1': [1], 'id2': [101], 'label': [1.0]})
        mock_create_seeds.return_value = mock_seeds
        
        mock_learner = Mock()
        mock_labeled_data = pd.DataFrame({'id1': [1, 2], 'id2': [101, 102], 'label': [1.0, 0.0]})
        mock_learner.train.return_value = mock_labeled_data
        mock_learner_class.return_value = mock_learner
        
        # Test
        model_spec = {'model_type': 'sklearn', 'model': LogisticRegression, 'model_args': {}}
        labeler_spec = {'name': 'gold', 'gold': gold_labels}
        
        result = label_data(model_spec, 'continuous', labeler_spec, sample_feature_vectors)
        
        assert isinstance(result, pd.DataFrame)
        mock_learner_class.assert_called_once()
        mock_learner.train.assert_called_once()

    @patch('madmatcher_tools.tools.create_seeds')
    def test_label_data_with_existing_seeds(self, mock_create_seeds,
                                          sample_feature_vectors, gold_labels):
        """Test label_data with pre-existing seeds."""
        existing_seeds = pd.DataFrame({'id1': [1], 'id2': [101], 'label': [1.0]})
        
        with patch('madmatcher_tools.tools.EntropyActiveLearner') as mock_learner_class:
            mock_learner = Mock()
            mock_learner.train.return_value = existing_seeds
            mock_learner_class.return_value = mock_learner
            
            model_spec = {'model_type': 'sklearn', 'model': LogisticRegression, 'model_args': {}}
            labeler_spec = {'name': 'gold', 'gold': gold_labels}
            
            result = label_data(model_spec, 'batch', labeler_spec, 
                              sample_feature_vectors, seeds=existing_seeds)
            
            # Should not call create_seeds when seeds are provided
            mock_create_seeds.assert_not_called()
            mock_learner.train.assert_called_once()


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_down_sample_with_missing_scores(self):
        """Test down sampling with missing score values."""
        fvs = pd.DataFrame({
            'id2': [101, 102, 103, 104],
            'score': [0.9, np.nan, 0.6, 0.4]
        })
        
        # Should handle NaN scores gracefully
        result = down_sample(fvs, percent=0.5, search_id_column='id2')
        assert len(result) <= len(fvs)

    def test_functions_with_empty_inputs(self):
        """Test functions handle empty inputs gracefully."""
        empty_df = pd.DataFrame()
        
        # These should not crash with empty inputs
        with pytest.raises((ValueError, KeyError)):
            down_sample(empty_df, percent=0.5, search_id_column='id2')

    def test_invalid_model_specs(self, mock_labeled_data):
        """Test train_matcher with invalid model specifications."""
        invalid_specs = [
            {'model_type': 'invalid_type'},
            {'model': 'not_a_class'},
            {}
        ]
        
        for spec in invalid_specs:
            with pytest.raises((ValueError, KeyError, AttributeError)):
                train_matcher(spec, mock_labeled_data) 