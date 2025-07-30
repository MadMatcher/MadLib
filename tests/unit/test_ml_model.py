import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from MadLib._internal.ml_model import SKLearnModel, SparkMLModel
from unittest.mock import MagicMock, patch

class DummyEstimator(BaseEstimator):
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y):
        self.is_fitted_ = True
        return self
    def predict(self, X):
        return np.zeros(X.shape[0])
    def predict_proba(self, X):
        return np.ones((X.shape[0], 2)) * 0.5

# Create a mock Transformer class
MockTransformer = type('MockTransformer', (), {})

class DummySparkTransformer(MockTransformer):
    def __init__(self, **kwargs):
        pass
    def getPredictionCol(self):
        return 'prediction'
    def getProbabilityCol(self):
        return 'probability'
    def setFeaturesCol(self, col):
        return self
    def transform(self, df):
        return df
    def fit(self, df):
        return self

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'features': [[1.0, 2.0], [3.0, 4.0]],
        'label': [0, 1]
    })

def test_init_with_unfitted_model():
    model = SKLearnModel(DummyEstimator)
    assert model._trained_model is None
    assert model._model == DummyEstimator
    assert isinstance(model._model_args, dict)

def test_init_with_fitted_model():
    est = DummyEstimator().fit(np.array([[1,2],[3,4]]), np.array([0,1]))
    model = SKLearnModel(est)
    assert model._trained_model is not None
    assert model._model == DummyEstimator

def test_get_model_and_params_dict():
    model = SKLearnModel(DummyEstimator, nan_fill=0.0, use_floats=False, execution='local', foo=1)
    m = model.get_model()
    assert isinstance(m, DummyEstimator)
    params = model.params_dict()
    assert 'model' in params and 'nan_fill' in params and 'model_args' in params

def test_make_feature_matrix_and_allocate_buffer():
    model = SKLearnModel(DummyEstimator)
    arr = [[1.0, 2.0], [3.0, 4.0]]
    X = model._make_feature_matrix(arr)
    assert X.shape == (2, 2)
    # Test empty input
    assert model._make_feature_matrix([]) is None
    # Test buffer allocation
    buf = model._allocate_buffer(2, 2)
    assert buf.shape == (2, 2)

def test_no_threads():
    model = SKLearnModel(DummyEstimator)
    model._no_threads()  # Should not crash

def test_predict_and_train(sample_df):
    model = SKLearnModel(DummyEstimator)
    model.train(sample_df, 'features', 'label')
    out = model.predict(sample_df, 'features', 'pred')
    assert 'pred' in out.columns
    # Only test the local (pandas) code path for coverage

def test_prediction_conf_and_entropy(sample_df):
    model = SKLearnModel(DummyEstimator)
    model.train(sample_df, 'features', 'label')
    # Only test the local (pandas) code path for coverage
    # This avoids the Spark code path that expects a Spark DataFrame
    # and .withColumn method
    # The Spark path is just a wrapper and is covered elsewhere
    # Test error if not trained
    model2 = SKLearnModel(DummyEstimator)
    with pytest.raises(RuntimeError):
        model2.prediction_conf(sample_df, 'features', 'conf')
    with pytest.raises(RuntimeError):
        model2.entropy(sample_df, 'features', 'ent')

# SparkMLModel tests
def test_spark_ml_model_init_with_unfitted_model():
    model = SparkMLModel(DummySparkTransformer)
    assert model._trained_model is None
    assert model._model == DummySparkTransformer
    assert isinstance(model._model_args, dict)

def test_spark_ml_model_init_with_fitted_model():
    transformer = DummySparkTransformer()
    # Mock the Transformer import
    with patch('MadLib._internal.ml_model.Transformer', MockTransformer):
        model = SparkMLModel(transformer)
        assert model._trained_model is not None
        assert model._model == DummySparkTransformer

def test_spark_ml_model_get_model_and_params_dict():
    model = SparkMLModel(DummySparkTransformer, nan_fill=0.0, foo=1)
    m = model.get_model()
    assert isinstance(m, DummySparkTransformer)
    params = model.params_dict()
    assert 'model' in params and 'model_args' in params

def test_spark_ml_model_properties():
    model = SparkMLModel(DummySparkTransformer)
    assert model.nan_fill == 0.0
    assert model.use_vectors is True
    assert model.use_floats is False
    assert model.trained_model is None

def test_spark_ml_model_predict_with_pandas_df():
    model = SparkMLModel(DummySparkTransformer)
    # Mock SparkSession
    with patch('MadLib._internal.ml_model.SparkSession') as mock_spark_session:
        mock_spark = MagicMock()
        mock_spark_session.builder.getOrCreate.return_value = mock_spark
        mock_spark.createDataFrame.return_value = MagicMock()
        
        # Mock convert_to_vector
        with patch('MadLib._internal.ml_model.convert_to_vector') as mock_convert:
            mock_convert.return_value = MagicMock()
            
            # Mock F.col
            with patch('MadLib._internal.ml_model.F') as mock_F:
                mock_F.col.return_value = MagicMock()
                mock_F.col.return_value.alias.return_value = MagicMock()
                
                # Test with pandas DataFrame
                df = pd.DataFrame({'features': [[1, 2], [3, 4]]})
                model._trained_model = DummySparkTransformer()
                result = model.predict(df, 'features', 'pred')
                assert result is not None

def test_spark_ml_model_predict_without_trained_model():
    model = SparkMLModel(DummySparkTransformer)
    df = pd.DataFrame({'features': [[1, 2], [3, 4]]})
    with pytest.raises(RuntimeError):
        model.predict(df, 'features', 'pred')

def test_spark_ml_model_prediction_conf_without_trained_model():
    model = SparkMLModel(DummySparkTransformer)
    df = pd.DataFrame({'features': [[1, 2], [3, 4]]})
    with pytest.raises(RuntimeError):
        model.prediction_conf(df, 'features', 'conf')

def test_spark_ml_model_entropy_without_trained_model():
    model = SparkMLModel(DummySparkTransformer)
    df = pd.DataFrame({'features': [[1, 2], [3, 4]]})
    with pytest.raises(RuntimeError):
        model.entropy(df, 'features', 'ent')

def test_spark_ml_model_entropy_component():
    model = SparkMLModel(DummySparkTransformer)
    # Test _entropy_component method
    with patch('MadLib._internal.ml_model.F') as mock_F:
        mock_get_item = MagicMock()
        mock_get_item.getItem.return_value = MagicMock()
        mock_F.when.return_value.otherwise.return_value = 0.5
        result = model._entropy_component(mock_get_item, 0)
        assert result == 0.5

def test_spark_ml_model_entropy_expr():
    model = SparkMLModel(DummySparkTransformer)
    # Test _entropy_expr method
    with patch('MadLib._internal.ml_model.F') as mock_F:
        mock_F.col.return_value = MagicMock()
        mock_F.when.return_value.otherwise.return_value = 0.5
        result = model._entropy_expr('probs', classes=2)
        assert result is not None

def test_sklearn_model_with_nan_fill():
    model = SKLearnModel(DummyEstimator, nan_fill=0.0)
    arr = [[1.0, np.nan], [3.0, 4.0]]
    X = model._make_feature_matrix(arr)
    assert X.shape == (2, 2)
    # Check that NaN was filled
    assert not np.isnan(X).any()

def test_sklearn_model_use_floats_false():
    model = SKLearnModel(DummyEstimator, use_floats=False)
    buf = model._allocate_buffer(2, 2)
    assert buf.dtype == np.float64

def test_sklearn_model_train_with_spark_df(sample_df):
    model = SKLearnModel(DummyEstimator)
    # Mock Spark DataFrame
    mock_spark_df = MagicMock()
    mock_spark_df.toPandas.return_value = sample_df
    
    with patch('MadLib._internal.ml_model.convert_to_array') as mock_convert:
        mock_convert.return_value = mock_spark_df
        model.train(mock_spark_df, 'features', 'label')
        assert model._trained_model is not None 