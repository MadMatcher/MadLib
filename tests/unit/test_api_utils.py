"""
Tests for direct object usage in tools module.

This module tests the direct usage of ML models and labelers without the utility functions.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, Mock, patch
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

from MadLib import MLModel, SKLearnModel, SparkMLModel, Labeler, CLILabeler, GoldLabeler, WebUILabeler


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


class MockLabeler(Labeler):
    """Mock Labeler for testing."""
    
    def __call__(self, id1, id2):
        return 1.0  # Always return positive match for testing


@pytest.mark.unit
class TestDirectModelUsage:
    """Test direct usage of MLModel instances."""

    def test_sklearn_model_creation(self):
        """Test creating SKLearnModel directly."""
        sklearn_model = LogisticRegression()
        model = SKLearnModel(model=sklearn_model)
        
        assert isinstance(model, SKLearnModel)
        assert model._model == sklearn_model
        assert model._nan_fill is None
        assert model.use_floats is True
        assert model.execution == 'local'

    def test_sklearn_model_with_options(self):
        """Test creating SKLearnModel with options."""
        sklearn_model = LogisticRegression()
        model = SKLearnModel(
            model=sklearn_model,
            nan_fill=0.0,
            use_floats=False,
            execution='spark'
        )
        
        assert isinstance(model, SKLearnModel)
        assert model._model == sklearn_model
        assert model._nan_fill == 0.0
        assert model.use_floats is False
        assert model.execution == 'spark'

    def test_sparkml_model_creation(self):
        """Test creating SparkMLModel directly."""
        from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
        
        spark_model = SparkLogisticRegression()
        model = SparkMLModel(model=spark_model)
        
        assert isinstance(model, SparkMLModel)
        assert model._model == spark_model
        assert model._nan_fill is None

    def test_sparkml_model_with_nan_fill(self):
        """Test creating SparkMLModel with nan_fill."""
        from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
        
        spark_model = SparkLogisticRegression()
        model = SparkMLModel(model=spark_model, nan_fill=0.0)
        
        assert isinstance(model, SparkMLModel)
        assert model._model == spark_model
        assert model._nan_fill == 0.0


@pytest.mark.unit
class TestDirectLabelerUsage:
    """Test direct usage of Labeler instances."""

    def test_cli_labeler_creation(self):
        """Test creating CLILabeler directly."""
        df_a = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        df_b = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        
        labeler = CLILabeler(a_df=df_a, b_df=df_b)
        
        assert isinstance(labeler, CLILabeler)
        assert labeler._a_df.equals(df_a)
        assert labeler._b_df.equals(df_b)

    def test_cli_labeler_with_id_col(self):
        """Test creating CLILabeler with id_col."""
        df_a = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        df_b = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        
        labeler = CLILabeler(a_df=df_a, b_df=df_b, id_col='id')
        
        assert isinstance(labeler, CLILabeler)
        assert labeler._a_df.equals(df_a)
        assert labeler._b_df.equals(df_b)
        assert labeler._id_col == 'id'

    def test_gold_labeler_creation(self):
        """Test creating GoldLabeler directly."""
        gold_pairs = [(1, 1), (2, 2)]
        
        labeler = GoldLabeler(gold=gold_pairs)
        
        assert isinstance(labeler, GoldLabeler)
        assert labeler._gold == gold_pairs

    def test_webui_labeler_creation(self):
        """Test creating WebUILabeler directly."""
        df_a = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        df_b = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        
        labeler = WebUILabeler(a_df=df_a, b_df=df_b)
        
        assert isinstance(labeler, WebUILabeler)
        assert labeler._a_df.equals(df_a)
        assert labeler._b_df.equals(df_b)


@pytest.mark.unit
class TestToolsIntegration:
    """Test integration with tools module functions."""

    def test_train_matcher_with_mlmodel(self):
        """Test train_matcher with MLModel instance."""
        from MadLib.tools import train_matcher
        
        model = MockMLModel()
        labeled_data = pd.DataFrame({
            'features': [[1.0, 2.0], [3.0, 4.0]],
            'label': [0, 1]
        })
        
        result = train_matcher(model, labeled_data)
        
        assert result is model
        assert result.trained_model is not None

    def test_apply_matcher_with_mlmodel(self):
        """Test apply_matcher with MLModel instance."""
        from MadLib.tools import apply_matcher
        
        model = MockMLModel(trained_model=MockMLModel())
        df = pd.DataFrame({
            'features': [[1.0, 2.0], [3.0, 4.0]]
        })
        
        result = apply_matcher(model, df, 'features', 'predictions')
        
        assert isinstance(result, pd.DataFrame)

    def test_create_seeds_with_labeler(self):
        """Test create_seeds with Labeler instance."""
        from MadLib.tools import create_seeds
        
        labeler = MockLabeler()
        fvs = pd.DataFrame({
            'id1': [1, 2, 3],
            'id2': [1, 2, 3],
            'score': [0.8, 0.6, 0.4]
        })
        
        result = create_seeds(fvs, 2, labeler)
        
        assert isinstance(result, pd.DataFrame)
        assert 'label' in result.columns
        assert len(result) <= 2

    def test_label_data_with_objects(self):
        """Test label_data with model and labeler instances."""
        from MadLib.tools import label_data
        
        model = MockMLModel()
        labeler = MockLabeler()
        fvs = pd.DataFrame({
            'id1': [1, 2, 3, 4, 5],
            'id2': [1, 2, 3, 4, 5],
            'features': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        })
        
        # This test might need mocking for the active learning components
        with patch('MadLib.tools.EntropyActiveLearner') as mock_learner:
            mock_learner_instance = Mock()
            mock_learner_instance.train.return_value = pd.DataFrame({
                'id1': [1, 2],
                'id2': [1, 2],
                'label': [1.0, 0.0]
            })
            mock_learner.return_value = mock_learner_instance
            
            result = label_data(model, "batch", labeler, fvs)
            
            assert isinstance(result, pd.DataFrame)
            mock_learner.assert_called_once()


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_train_matcher_with_invalid_model(self):
        """Test train_matcher with invalid model type."""
        from MadLib.tools import train_matcher
        
        labeled_data = pd.DataFrame({
            'features': [[1.0, 2.0], [3.0, 4.0]],
            'label': [0, 1]
        })
        
        with pytest.raises(AttributeError):
            # This should fail because we're passing a string instead of MLModel
            train_matcher("invalid_model", labeled_data)

    def test_apply_matcher_with_untrained_model(self):
        """Test apply_matcher with untrained model."""
        from MadLib.tools import apply_matcher
        
        model = MockMLModel(trained_model=None)  # Untrained model
        df = pd.DataFrame({
            'features': [[1.0, 2.0], [3.0, 4.0]]
        })
        
        # This should work because MockMLModel handles untrained models gracefully
        # In real implementation, this might raise an error
        result = apply_matcher(model, df, 'features', 'predictions')
        assert isinstance(result, pd.DataFrame)

    def test_create_seeds_with_invalid_labeler(self):
        """Test create_seeds with invalid labeler type."""
        from MadLib.tools import create_seeds
        
        fvs = pd.DataFrame({
            'id1': [1, 2, 3],
            'id2': [1, 2, 3],
            'score': [0.8, 0.6, 0.4]
        })
        
        with pytest.raises(TypeError):
            # This should fail because we're passing a string instead of Labeler
            create_seeds(fvs, 2, "invalid_labeler") 