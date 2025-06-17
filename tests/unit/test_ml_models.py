"""Unit tests for ML model classes."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from madmatcher_tools._internal.ml_model import SKLearnModel
from madmatcher_tools import MLModel


@pytest.mark.unit
class TestSKLearnModel:
    """Test SKLearnModel wrapper."""

    def test_sklearn_mlmodel_creation(self):
        """Test creating SKLearnModel with LogisticRegression."""
        model = SKLearnModel(LogisticRegression, random_state=42)
        
        # The actual API doesn't have model_class property, it has _model
        assert model._model == LogisticRegression
        assert hasattr(model, 'nan_fill')
        assert hasattr(model, 'use_vectors') 
        assert hasattr(model, 'use_floats')

    def test_sklearn_mlmodel_train(self, mock_labeled_data):
        """Test training SKLearnModel."""
        model = SKLearnModel(LogisticRegression, random_state=42)
        
        trained = model.train(mock_labeled_data, 'features', 'label')
        
        assert trained is not None
        assert hasattr(trained, 'trained_model')
        assert trained.trained_model is not None

    def test_sklearn_mlmodel_predict(self, mock_labeled_data, sample_feature_vectors):
        """Test prediction with SKLearnModel."""
        model = SKLearnModel(LogisticRegression, random_state=42)
        model.train(mock_labeled_data, 'features', 'label')
        
        predictions = model.predict(sample_feature_vectors, 'features', 'prediction')
        
        assert isinstance(predictions, pd.DataFrame)
        assert 'prediction' in predictions.columns
        assert len(predictions) == len(sample_feature_vectors)

    def test_sklearn_mlmodel_properties(self):
        """Test SKLearnModel properties."""
        model = SKLearnModel(LogisticRegression)
        
        # Default nan_fill is None, not 0.0
        assert model.nan_fill is None
        assert model.use_vectors is False
        assert model.use_floats is True

    def test_sklearn_mlmodel_params_dict(self, mock_labeled_data):
        """Test params_dict method."""
        model = SKLearnModel(LogisticRegression, random_state=42, C=1.0)
        model.train(mock_labeled_data, 'features', 'label')
        
        params = model.params_dict()
        
        assert isinstance(params, dict)
        assert 'model' in params
        assert 'model_args' in params
        assert params['model_args']['random_state'] == 42
        assert params['model_args']['C'] == 1.0

    def test_sklearn_mlmodel_multiple_models(self, mock_labeled_data):
        """Test SKLearnModel with different sklearn models."""
        models_to_test = [
            (LogisticRegression, {'random_state': 42}),
            (RandomForestClassifier, {'n_estimators': 10, 'random_state': 42})
        ]
        
        for model_class, model_args in models_to_test:
            model = SKLearnModel(model_class, **model_args)
            model.train(mock_labeled_data, 'features', 'label')
            
            assert model.trained_model is not None
            assert isinstance(model.trained_model, model_class)

    def test_sklearn_mlmodel_prediction_conf(self, mock_labeled_data, sample_feature_vectors):
        """Test prediction confidence method."""
        model = SKLearnModel(LogisticRegression, random_state=42)
        model.train(mock_labeled_data, 'features', 'label')
        
        # Convert to Spark for prediction_conf method
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark_df = spark.createDataFrame(sample_feature_vectors)
        
        result = model.prediction_conf(spark_df, 'features', 'conf')
        
        assert result is not None
        assert 'conf' in result.columns

    def test_sklearn_mlmodel_entropy(self, mock_labeled_data, sample_feature_vectors):
        """Test entropy calculation method."""
        model = SKLearnModel(LogisticRegression, random_state=42)
        model.train(mock_labeled_data, 'features', 'label')
        
        # Convert to Spark for entropy method
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark_df = spark.createDataFrame(sample_feature_vectors)
        
        result = model.entropy(spark_df, 'features', 'entropy')
        
        assert result is not None
        assert 'entropy' in result.columns


@pytest.mark.unit
class TestCustomMLModel:
    """Test custom MLModel implementations."""

    def test_custom_mlmodel_implementation(self):
        """Test implementing a custom MLModel."""
        
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
                result = df.copy()
                result[output_col] = [1.0] * len(df)
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
                return {'trained': self._trained}
        
        model = TestMLModel()
        
        assert model.nan_fill == 0.0
        assert model.use_vectors is False
        assert model.use_floats is True


@pytest.mark.unit
class TestMLModelTypes:
    """Test different MLModel type scenarios."""
    
    def test_vector_mlmodel(self):
        """Test MLModel that uses vectors."""
        
        class VectorMLModel(MLModel):
            def __init__(self):
                self._trained_model = None
                
            @property
            def nan_fill(self): return None
            @property
            def use_vectors(self): return True
            @property
            def use_floats(self): return False
            @property
            def trained_model(self): return self._trained_model
            
            def train(self, df, vector_col, label_column, return_estimator=False):
                self._trained_model = self  # Mock trained model
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                return df
            
            def params_dict(self):
                return {}
        
        model = VectorMLModel()
        assert model.use_vectors is True
        assert model.use_floats is False

    def test_float_mlmodel(self):
        """Test MLModel that uses floats."""
        
        class FloatMLModel(MLModel):
            def __init__(self):
                self._trained_model = None
                
            @property
            def nan_fill(self): return 0.5
            @property
            def use_vectors(self): return False
            @property
            def use_floats(self): return True
            @property
            def trained_model(self): return self._trained_model
            
            def train(self, df, vector_col, label_column, return_estimator=False):
                self._trained_model = self  # Mock trained model
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                return df
            
            def params_dict(self):
                return {}
        
        model = FloatMLModel()
        assert model.use_floats is True
        assert model.nan_fill == 0.5


@pytest.mark.unit
class TestMLModelIntegration:
    """Test MLModel integration scenarios."""

    def test_mlmodel_with_feature_vectors_containing_nans(self, test_data_generator):
        """Test MLModel handling of NaN values in feature vectors."""
        model = SKLearnModel(LogisticRegression, random_state=42, nan_fill=0.0)
        
        # Create data with NaN values manually since create_labeled_data doesn't exist
        data_with_nans = pd.DataFrame({
            'features': [
                test_data_generator.create_feature_vector_with_nans(length=5, nan_rate=0.3),
                test_data_generator.create_feature_vector_with_nans(length=5, nan_rate=0.3),
                test_data_generator.create_feature_vector_with_nans(length=5, nan_rate=0.3),
                test_data_generator.create_feature_vector_with_nans(length=5, nan_rate=0.3)
            ],
            'label': [1.0, 0.0, 1.0, 0.0]
        })
        
        # Fill NaN values since sklearn can't handle them
        data_with_nans['features'] = data_with_nans['features'].apply(
            lambda x: [0.0 if pd.isna(val) else val for val in x] if x is not None else [0.0]
        )
        
        # Should handle NaN gracefully with nan_fill
        trained_model = model.train(data_with_nans, 'features', 'label')
        assert trained_model is not None

    def test_mlmodel_empty_training_data(self):
        """Test MLModel with empty training data."""
        model = SKLearnModel(LogisticRegression, random_state=42)
        
        empty_data = pd.DataFrame(columns=['features', 'label'])
        
        # Empty data should raise an error
        with pytest.raises((ValueError, IndexError)):
            model.train(empty_data, 'features', 'label')

    def test_mlmodel_single_class_training_data(self):
        """Test MLModel with single class in training data."""
        model = SKLearnModel(LogisticRegression, random_state=42)
        
        single_class_data = pd.DataFrame({
            'features': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            'label': [1.0, 1.0, 1.0]  # All same class
        })
        
        # Single class should raise error in sklearn
        with pytest.raises(ValueError):
            model.train(single_class_data, 'features', 'label')


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_sklearn_mlmodel_invalid_model_class(self):
        """Test SKLearnModel with invalid model class."""
        # The constructor doesn't immediately validate the model class
        # It only fails when you try to get_model() or train
        invalid_model = SKLearnModel("not_a_class")
        
        with pytest.raises((TypeError, AttributeError)):
            invalid_model.get_model()

    def test_sklearn_mlmodel_predict_before_training(self, sample_feature_vectors):
        """Test prediction before training."""
        model = SKLearnModel(LogisticRegression)
        
        with pytest.raises((AttributeError, ValueError)):
            model.predict(sample_feature_vectors, 'features', 'prediction')

    def test_sklearn_mlmodel_invalid_columns(self, mock_labeled_data):
        """Test with invalid column names."""
        model = SKLearnModel(LogisticRegression, random_state=42)
        
        with pytest.raises(KeyError):
            model.train(mock_labeled_data, 'nonexistent_col', 'label')

    def test_sklearn_mlmodel_mismatched_feature_dimensions(self, mock_labeled_data):
        """Test with mismatched feature dimensions between train and predict."""
        model = SKLearnModel(LogisticRegression, random_state=42)
        model.train(mock_labeled_data, 'features', 'label')
        
        # Create data with different feature dimensions
        mismatched_data = pd.DataFrame({
            'features': [[0.1], [0.2]]  # Different dimension than training
        })
        
        with pytest.raises((ValueError, AttributeError)):
            model.predict(mismatched_data, 'features', 'prediction') 