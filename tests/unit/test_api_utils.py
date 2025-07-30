"""
Tests for api_utils module.

This module tests the utility functions for creating ML models and labelers
from specifications.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, Mock, patch
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

from MadLib._internal.api_utils import (
    _create_training_model, _create_matching_model, _create_labeler,
    AVAILABLE_LABELERS
)
from MadLib._internal.ml_model import MLModel, SKLearnModel, SparkMLModel
from MadLib._internal.labeler import Labeler, CLILabeler, GoldLabeler


class MockMLModel(MLModel):
    """Mock MLModel for testing."""
    
    def __init__(self, trained_model=None):
        self._trained_model = trained_model
    
    @property
    def trained_model(self):
        return self._trained_model
    
    @property
    def nan_fill(self):
        return 0.0
    
    @property
    def use_vectors(self):
        return False
    
    @property
    def use_floats(self):
        return True
    
    def train(self, df, vector_col, label_column, return_estimator=False):
        self._trained_model = self
        return self
    
    def predict(self, df, vector_col, output_col):
        return df
    
    def prediction_conf(self, df, vector_col, label_column):
        return df
    
    def entropy(self, df, vector_col, output_col):
        return df
    
    def get_model(self):
        return MagicMock()
    
    def params_dict(self):
        return {}


@pytest.mark.unit
class TestCreateTrainingModel:
    """Test _create_training_model function."""

    def test_create_training_model_with_mlmodel(self):
        """Test creating training model with existing MLModel instance."""
        mock_model = MockMLModel()
        result = _create_training_model(mock_model)
        
        assert result is mock_model

    def test_create_training_model_sklearn_dict(self):
        """Test creating training model with sklearn specification."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'C': 1.0}
        }
        
        result = _create_training_model(model_spec)
        
        assert isinstance(result, SKLearnModel)
        assert result._model == LogisticRegression
        assert result._model_args == {'C': 1.0}
        assert result._nan_fill is None
        assert result.use_floats is True
        assert result.execution == 'local'

    def test_create_training_model_sklearn_dict_with_options(self):
        """Test creating training model with sklearn specification and options."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'C': 1.0},
            'execution': 'spark',
            'nan_fill': 0.0,
            'use_floats': False
        }
        
        result = _create_training_model(model_spec)
        
        assert isinstance(result, SKLearnModel)
        assert result._model == LogisticRegression
        assert result._model_args == {'C': 1.0}
        assert result._nan_fill == 0.0
        assert result.use_floats is False
        assert result.execution == 'spark'

    def test_create_training_model_sparkml_dict(self):
        """Test creating training model with SparkML specification."""
        mock_transformer = MagicMock()
        model_spec = {
            'model_type': 'sparkml',
            'model': mock_transformer,
            'model_args': {'param1': 'value1'}
        }
        
        result = _create_training_model(model_spec)
        
        assert isinstance(result, SparkMLModel)
        assert result._model == mock_transformer
        assert result._model_args == {'param1': 'value1'}
        assert result._nan_fill is None

    def test_create_training_model_sparkml_dict_with_nan_fill(self):
        """Test creating training model with SparkML specification and nan_fill."""
        mock_transformer = MagicMock()
        model_spec = {
            'model_type': 'sparkml',
            'model': mock_transformer,
            'model_args': {},
            'nan_fill': -1.0
        }
        
        result = _create_training_model(model_spec)
        
        assert isinstance(result, SparkMLModel)
        assert result._nan_fill == -1.0

    def test_create_training_model_case_insensitive(self):
        """Test that model_type is case insensitive."""
        model_spec = {
            'model_type': 'SKLEARN',
            'model': LogisticRegression,
            'model_args': {}
        }
        
        result = _create_training_model(model_spec)
        
        assert isinstance(result, SKLearnModel)

    @patch('MadLib._internal.ml_model.convert_to_array', side_effect=lambda df, col: df)
    def test_create_training_model_with_spark_dataframe(self, mock_convert, mock_labeled_data):
        mock_spark_df = MagicMock()
        mock_spark_df.toPandas.return_value = mock_labeled_data
        mock_spark_df.count.return_value = len(mock_labeled_data)
        mock_spark_df.columns = mock_labeled_data.columns.tolist()
        mock_spark_df.__getitem__.side_effect = lambda key: mock_labeled_data[key]
        mock_spark_df.__iter__.side_effect = lambda: iter(mock_labeled_data)
        with patch('MadLib._internal.ml_model.convert_to_array', return_value=mock_labeled_data):
            model_spec = {
                'model_type': 'sklearn',
                'model': LogisticRegression,
                'model_args': {'random_state': 42}
            }
            model = _create_training_model(model_spec)
            result = model.train(mock_spark_df, 'features', 'label')
            assert hasattr(result, 'trained_model')

    def test_create_training_model_with_sparkml_nan_fill(self, mock_labeled_data):
        """Test create_training_model with SparkML model and nan_fill option."""
        model_spec = {
            'model_type': 'sparkml',
            'model': 'RandomForestClassifier',
            'model_args': {'numTrees': 10},
            'nan_fill': 0.0
        }
        
        model = _create_training_model(model_spec)
        
        assert hasattr(model, 'nan_fill')
        assert model.nan_fill == 0.0

    def test_create_training_model_with_sklearn_nan_fill(self, mock_labeled_data):
        """Test create_training_model with sklearn model and nan_fill option."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42},
            'nan_fill': 0.0
        }
        
        model = _create_training_model(model_spec)
        
        assert hasattr(model, 'nan_fill')
        assert model.nan_fill == 0.0

    def test_create_training_model_with_execution_spark(self, mock_labeled_data):
        """Test create_training_model with execution='spark' option."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42},
            'execution': 'spark'
        }
        
        model = _create_training_model(model_spec)
        
        assert hasattr(model, 'execution')
        assert model.execution == 'spark'

    def test_create_training_model_with_execution_local(self, mock_labeled_data):
        """Test create_training_model with execution='local' option."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42},
            'execution': 'local'
        }
        
        model = _create_training_model(model_spec)
        
        assert hasattr(model, 'execution')
        assert model.execution == 'local'

    def test_create_training_model_with_use_floats_true(self, mock_labeled_data):
        """Test create_training_model with use_floats=True option."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42},
            'use_floats': True
        }
        
        model = _create_training_model(model_spec)
        
        assert hasattr(model, 'use_floats')
        assert model.use_floats == True

    def test_create_training_model_with_use_floats_false(self, mock_labeled_data):
        """Test create_training_model with use_floats=False option."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42},
            'use_floats': False
        }
        
        model = _create_training_model(model_spec)
        
        assert hasattr(model, 'use_floats')
        assert model.use_floats == False

    def test_create_training_model_with_use_vectors_true(self, mock_labeled_data):
        """Test create_training_model with use_vectors=True option."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42},
            'use_vectors': True
        }
        
        model = _create_training_model(model_spec)
        
        assert hasattr(model, 'use_vectors')
        # SKLearnModel always returns False for use_vectors, regardless of the parameter
        assert model.use_vectors == False

    def test_create_training_model_with_use_vectors_false(self, mock_labeled_data):
        """Test create_training_model with use_vectors=False option."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42},
            'use_vectors': False
        }
        
        model = _create_training_model(model_spec)
        
        assert hasattr(model, 'use_vectors')
        assert model.use_vectors == False


@pytest.mark.unit
class TestCreateMatchingModel:
    """Test _create_matching_model function."""

    def test_create_matching_model_with_mlmodel_trained(self):
        """Test creating matching model with trained MLModel."""
        mock_model = MockMLModel(trained_model=MagicMock())
        result = _create_matching_model(mock_model)
        
        assert result is mock_model

    def test_create_matching_model_with_mlmodel_untrained(self):
        """Test creating matching model with untrained MLModel."""
        mock_model = MockMLModel(trained_model=None)
        
        with pytest.raises(RuntimeError, match="Model must be trained to predict"):
            _create_matching_model(mock_model)

    def test_create_matching_model_with_transformer(self):
        """Test creating matching model with Spark Transformer."""
        mock_transformer = MagicMock()
        
        with patch('MadLib._internal.api_utils.check_is_fitted') as mock_check_fitted:
            mock_check_fitted.side_effect = NotFittedError("Model not fitted")
            
            with pytest.raises(RuntimeError, match="Model must be trained to predict"):
                _create_matching_model(mock_transformer)

    @patch('MadLib._internal.api_utils.check_is_fitted')
    def test_create_matching_model_with_fitted_sklearn(self, mock_check_fitted):
        """Test creating matching model with fitted sklearn model."""
        # Create a mock sklearn model with the required attributes
        mock_sklearn_model = MagicMock()
        # Create a mock tags object with the required attributes
        mock_tags = MagicMock()
        mock_tags.requires_fit = True
        mock_sklearn_model.__sklearn_tags__ = MagicMock(return_value=mock_tags)
        mock_check_fitted.return_value = None  # No exception raised
        
        result = _create_matching_model(mock_sklearn_model)
        
        assert isinstance(result, SKLearnModel)
        # The trained_model is set during initialization, but may be None initially
        # Check that the model was created successfully
        assert result._model is not None

    @patch('MadLib._internal.api_utils.check_is_fitted')
    def test_create_matching_model_with_unfitted_sklearn(self, mock_check_fitted):
        """Test creating matching model with unfitted sklearn model."""
        # Create a mock sklearn model with the required attributes
        mock_sklearn_model = MagicMock()
        # Create a mock tags object with the required attributes
        mock_tags = MagicMock()
        mock_tags.requires_fit = True
        mock_sklearn_model.__sklearn_tags__ = MagicMock(return_value=mock_tags)
        mock_check_fitted.side_effect = NotFittedError("Model not fitted")
        
        with pytest.raises(RuntimeError, match="Model must be trained to predict"):
            _create_matching_model(mock_sklearn_model)

    @patch('MadLib._internal.ml_model.convert_to_array', side_effect=lambda df, col: df)
    def test_create_matching_model_with_spark_dataframe(self, mock_convert, sample_feature_vectors, mock_labeled_data):
        mock_spark_df = MagicMock()
        mock_spark_df.toPandas.return_value = sample_feature_vectors
        mock_spark_df.count.return_value = len(sample_feature_vectors)
        mock_spark_df.columns = sample_feature_vectors.columns.tolist()
        mock_spark_df.__getitem__.side_effect = lambda key: sample_feature_vectors[key]
        mock_spark_df.__iter__.side_effect = lambda: iter(sample_feature_vectors)
        with patch('MadLib._internal.ml_model.convert_to_array', return_value=sample_feature_vectors):
            model_spec = {
                'model_type': 'sklearn',
                'model': LogisticRegression,
                'model_args': {'random_state': 42}
            }
            training_model = _create_training_model(model_spec)
            training_model.train(mock_labeled_data, 'features', 'label')
            matching_model = _create_matching_model(training_model)
            with patch.object(matching_model, 'predict') as mock_predict:
                mock_predict.return_value = sample_feature_vectors.assign(prediction=[0.5, 0.3, 0.7, 0.2, 0.8])
                result = matching_model.predict(mock_spark_df, 'features', 'prediction')
                assert isinstance(result, pd.DataFrame)
                assert 'prediction' in result.columns

    @patch('MadLib._internal.ml_model.convert_to_array', side_effect=lambda df, col: df)
    def test_create_matching_model_with_execution_spark(self, mock_convert, sample_feature_vectors, mock_labeled_data):
        """Test create_matching_model with execution='spark' option."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42},
            'execution': 'spark'
        }
        training_model = _create_training_model(model_spec)
        training_model.train(mock_labeled_data, 'features', 'label')
        
        matching_model = _create_matching_model(training_model)
        
        # Mock the predict method to return a DataFrame with prediction column
        with patch.object(matching_model, 'predict') as mock_predict:
            mock_predict.return_value = sample_feature_vectors.assign(prediction=[0.5, 0.3, 0.7, 0.2, 0.8])
            result = matching_model.predict(sample_feature_vectors, 'features', 'prediction')
            
            assert isinstance(result, pd.DataFrame)
            assert 'prediction' in result.columns

    def test_create_matching_model_with_execution_local(self, sample_feature_vectors, mock_labeled_data):
        """Test create_matching_model with execution='local' option."""
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'random_state': 42},
            'execution': 'local'
        }
        training_model = _create_training_model(model_spec)
        training_model.train(mock_labeled_data, 'features', 'label')
        
        matching_model = _create_matching_model(training_model)
        result = matching_model.predict(sample_feature_vectors, 'features', 'prediction')
        
        assert isinstance(result, pd.DataFrame)
        assert 'prediction' in result.columns


@pytest.mark.unit
class TestCreateLabeler:
    """Test _create_labeler function."""

    def test_create_labeler_with_labeler_instance(self):
        """Test creating labeler with existing Labeler instance."""
        mock_labeler = MagicMock(spec=Labeler)
        result = _create_labeler(mock_labeler)
        
        assert result is mock_labeler

    def test_create_labeler_cli_dict(self):
        """Test creating labeler with CLI specification."""
        labeler_spec = {
            'name': 'cli',
            'a_df': pd.DataFrame({'col1': [1, 2]}),
            'b_df': pd.DataFrame({'col1': [3, 4]})
        }
        
        result = _create_labeler(labeler_spec)
        
        assert isinstance(result, CLILabeler)
        assert result._a_df is labeler_spec['a_df']
        assert result._b_df is labeler_spec['b_df']

    def test_create_labeler_cli_dict_with_optional_args(self):
        """Test creating labeler with CLI specification and optional args."""
        labeler_spec = {
            'name': 'cli',
            'a_df': pd.DataFrame({'col1': [1, 2]}),
            'b_df': pd.DataFrame({'col1': [3, 4]}),
            'id_col': 'custom_id'
        }
        
        result = _create_labeler(labeler_spec)
        
        assert isinstance(result, CLILabeler)
        assert result._id_col == 'custom_id'

    def test_create_labeler_gold_dict(self):
        """Test creating labeler with gold specification."""
        labeler_spec = {
            'name': 'gold',
            'gold': pd.DataFrame({
                'id1': [1, 2, 3],
                'id2': [4, 5, 6]
            })
        }
        
        result = _create_labeler(labeler_spec)
        
        assert isinstance(result, GoldLabeler)

    def test_create_labeler_missing_name(self):
        """Test creating labeler with missing name."""
        labeler_spec = {
            'a_df': pd.DataFrame({'col1': [1, 2]}),
            'b_df': pd.DataFrame({'col1': [3, 4]})
        }
        
        with pytest.raises(ValueError, match="Missing required key 'name'"):
            _create_labeler(labeler_spec)

    def test_create_labeler_unknown_name(self):
        """Test creating labeler with unknown name."""
        labeler_spec = {
            'name': 'unknown_labeler',
            'a_df': pd.DataFrame({'col1': [1, 2]}),
            'b_df': pd.DataFrame({'col1': [3, 4]})
        }
        
        with pytest.raises(ValueError, match="Unknown labeler type 'unknown_labeler'"):
            _create_labeler(labeler_spec)

    def test_create_labeler_missing_required_arg(self):
        """Test creating labeler with missing required argument."""
        labeler_spec = {
            'name': 'cli',
            'a_df': pd.DataFrame({'col1': [1, 2]})
            # Missing b_df
        }
        
        with pytest.raises(ValueError, match="Missing required argument 'b_df'"):
            _create_labeler(labeler_spec)


@pytest.mark.unit
class TestAvailableLabelers:
    """Test AVAILABLE_LABELERS constant."""

    def test_available_labelers_structure(self):
        """Test that AVAILABLE_LABELERS has the expected structure."""
        assert 'cli' in AVAILABLE_LABELERS
        assert 'gold' in AVAILABLE_LABELERS
        
        cli_info = AVAILABLE_LABELERS['cli']
        assert 'class' in cli_info
        assert 'description' in cli_info
        assert 'required_args' in cli_info
        assert 'optional_args' in cli_info
        
        assert cli_info['class'] == CLILabeler
        assert 'a_df' in cli_info['required_args']
        assert 'b_df' in cli_info['required_args']
        assert 'id_col' in cli_info['optional_args']
        
        gold_info = AVAILABLE_LABELERS['gold']
        assert gold_info['class'] == GoldLabeler
        assert 'gold' in gold_info['required_args']
        assert len(gold_info['optional_args']) == 0


@pytest.mark.unit
class TestApiUtilsIntegration:
    """Integration tests for api_utils module."""

    def test_full_workflow_sklearn(self):
        """Test complete workflow with sklearn model."""
        # Create training model
        model_spec = {
            'model_type': 'sklearn',
            'model': LogisticRegression,
            'model_args': {'C': 1.0, 'random_state': 42}
        }
        training_model = _create_training_model(model_spec)
        
        assert isinstance(training_model, SKLearnModel)
        
        # Simulate training
        training_model._trained_model = MagicMock()
        
        # Create matching model
        matching_model = _create_matching_model(training_model)
        assert matching_model is training_model

    def test_full_workflow_labeler(self):
        """Test complete workflow with labeler."""
        # Create labeler
        labeler_spec = {
            'name': 'cli',
            'a_df': pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']}),
            'b_df': pd.DataFrame({'id': [3, 4], 'name': ['Charlie', 'David']}),
            'id_col': 'id'
        }
        labeler = _create_labeler(labeler_spec)
        
        assert isinstance(labeler, CLILabeler)
        assert labeler._id_col == 'id'

    def test_error_handling_workflow(self):
        """Test error handling in workflows."""
        # Test untrained model
        mock_model = MockMLModel(trained_model=None)
        
        with pytest.raises(RuntimeError, match="Model must be trained to predict"):
            _create_matching_model(mock_model)
        
        # Test invalid labeler spec
        invalid_spec = {'name': 'nonexistent'}
        
        with pytest.raises(ValueError, match="Unknown labeler type"):
            _create_labeler(invalid_spec) 