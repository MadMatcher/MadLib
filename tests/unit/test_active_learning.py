"""Unit tests for active learning modules."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, DoubleType

from madmatcher_tools._internal.active_learning.ent_active_learner import EntropyActiveLearner
from madmatcher_tools._internal.active_learning.cont_entropy_active_learner import (
    ContinuousEntropyActiveLearner, PQueueItem
)
from madmatcher_tools._internal.ml_model import MLModel
from madmatcher_tools._internal.labeler import Labeler


@pytest.mark.unit
class TestEntropyActiveLearner:
    """Test EntropyActiveLearner functionality."""

    def test_entropy_active_learner_creation(self, mock_labeled_data):
        """Test creating EntropyActiveLearner."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                self._trained_model = self
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                # Always use a real DataFrame if not a Spark DataFrame
                if hasattr(df, 'toPandas'):
                    result = df.toPandas()
                else:
                    result = pd.DataFrame({
                        '_id': [1, 2, 3],
                        'id1': [101, 102, 103],
                        'id2': [201, 202, 203],
                        'features': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                    })
                if not isinstance(result, pd.DataFrame):
                    result = pd.DataFrame({
                        '_id': [1, 2, 3],
                        'id1': [101, 102, 103],
                        'id2': [201, 202, 203],
                        'features': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                    })
                result[output_col] = 0.5
                return spark_session.createDataFrame(result)
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0 if id1 == id2 else 0.0

        model = TestMLModel()
        labeler = TestLabeler()
        
        learner = EntropyActiveLearner(model, labeler, batch_size=5, max_iter=10)
        
        assert learner._batch_size == 5
        assert learner._max_iter == 10
        assert learner._labeler == labeler
        assert isinstance(learner._model, TestMLModel)
        assert learner._model.nan_fill == model.nan_fill
        assert learner._model.use_vectors == model.use_vectors
        assert learner._model.use_floats == model.use_floats

    def test_entropy_active_learner_invalid_args(self):
        """Test EntropyActiveLearner with invalid arguments."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                return df
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0

        model = TestMLModel()
        labeler = TestLabeler()
        
        # Test invalid batch_size
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            EntropyActiveLearner(model, labeler, batch_size=0)
        
        # Test invalid max_iter
        with pytest.raises(ValueError, match="max_iter must be > 0"):
            EntropyActiveLearner(model, labeler, max_iter=0)

    def test_entropy_active_learner_get_pos_negative(self):
        """Test _get_pos_negative method."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                return df
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0

        model = TestMLModel()
        labeler = TestLabeler()
        learner = EntropyActiveLearner(model, labeler)
        
        # Test with mixed labels
        batch = pd.DataFrame({'label': [1.0, 0.0, 1.0, 0.0, 1.0]})
        pos, neg = learner._get_pos_negative(batch)
        assert pos == 3
        assert neg == 2
        
        # Test with all positive
        batch = pd.DataFrame({'label': [1.0, 1.0, 1.0]})
        pos, neg = learner._get_pos_negative(batch)
        assert pos == 3
        assert neg == 0
        
        # Test with all negative
        batch = pd.DataFrame({'label': [0.0, 0.0, 0.0]})
        pos, neg = learner._get_pos_negative(batch)
        assert pos == 0
        assert neg == 3

    def test_entropy_active_learner_select_training_vectors(self, spark_session):
        """Test _select_training_vectors method."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                # Always use a real DataFrame if not a Spark DataFrame
                if hasattr(df, 'toPandas'):
                    result = df.toPandas()
                else:
                    result = pd.DataFrame({
                        '_id': [1, 2, 3],
                        'id1': [101, 102, 103],
                        'id2': [201, 202, 203],
                        'features': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                    })
                if not isinstance(result, pd.DataFrame):
                    result = pd.DataFrame({
                        '_id': [1, 2, 3],
                        'id1': [101, 102, 103],
                        'id2': [201, 202, 203],
                        'features': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                    })
                result[output_col] = 0.5
                return spark_session.createDataFrame(result)
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0

        model = TestMLModel()
        labeler = TestLabeler()
        learner = EntropyActiveLearner(model, labeler)
        
        # Create test data
        data = [(1, 101, 201, [0.1, 0.2]), (2, 102, 202, [0.3, 0.4]), (3, 103, 203, [0.5, 0.6])]
        schema = StructType([
            StructField("_id", IntegerType(), False),
            StructField("id1", IntegerType(), False),
            StructField("id2", IntegerType(), False),
            StructField("features", ArrayType(DoubleType()), False)
        ])
        fvs = spark_session.createDataFrame(data, schema)
        
        # Test selection
        ids = [1, 3]
        result = learner._select_training_vectors(fvs, ids)
        assert result.count() == 2
        result_ids = [row._id for row in result.collect()]
        assert set(result_ids) == {1, 3}

    @patch('madmatcher_tools._internal.active_learning.ent_active_learner.SparkSession')
    def test_entropy_active_learner_label_everything(self, mock_spark_session, spark_session):
        """Test _label_everything method."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                self._trained_model = self
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                # Always use a real DataFrame if not a Spark DataFrame
                if hasattr(df, 'toPandas'):
                    result = df.toPandas()
                else:
                    result = pd.DataFrame({
                        '_id': [1, 2, 3],
                        'id1': [101, 102, 103],
                        'id2': [201, 202, 203],
                        'features': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                    })
                if not isinstance(result, pd.DataFrame):
                    result = pd.DataFrame({
                        '_id': [1, 2, 3],
                        'id1': [101, 102, 103],
                        'id2': [201, 202, 203],
                        'features': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                    })
                result[output_col] = 0.5
                return spark_session.createDataFrame(result)
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0 if id1 == id2 else 0.0

        mock_spark_session.builder.getOrCreate.return_value = spark_session
        
        model = TestMLModel()
        labeler = TestLabeler()
        learner = EntropyActiveLearner(model, labeler)
        
        # Create test data
        data = [(1, 101, 201, [0.1, 0.2]), (2, 102, 202, [0.3, 0.4])]
        schema = StructType([
            StructField("_id", IntegerType(), False),
            StructField("id1", IntegerType(), False),
            StructField("id2", IntegerType(), False),
            StructField("features", ArrayType(DoubleType()), False)
        ])
        fvs = spark_session.createDataFrame(data, schema)
        
        result = learner._label_everything(fvs)
        
        assert isinstance(result, pd.DataFrame)
        assert 'label' in result.columns
        assert 'labeled_in_iteration' in result.columns
        assert len(result) == 2
        assert result['labeled_in_iteration'].iloc[0] == -2

    @patch('madmatcher_tools._internal.active_learning.ent_active_learner.SparkSession')
    @patch('madmatcher_tools._internal.active_learning.ent_active_learner.persisted')
    def test_entropy_active_learner_train_small_dataset(self, mock_persisted, mock_spark_session, spark_session):
        """Test train method with small dataset that triggers label_everything."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                self._trained_model = self
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                result = df.copy()
                result[output_col] = 0.5
                return result
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0 if id1 == id2 else 0.0

        mock_spark_session.builder.getOrCreate.return_value = spark_session
        
        # Mock persisted context manager
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=Mock(count=Mock(return_value=5)))
        mock_context.__exit__ = Mock(return_value=None)
        mock_persisted.return_value = mock_context
        
        model = TestMLModel()
        labeler = TestLabeler()
        learner = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
        learner._terminate_if_label_everything = True
        
        # Create test data
        data = [(1, 101, 201, [0.1, 0.2]), (2, 102, 202, [0.3, 0.4])]
        schema = StructType([
            StructField("_id", IntegerType(), False),
            StructField("id1", IntegerType(), False),
            StructField("id2", IntegerType(), False),
            StructField("features", ArrayType(DoubleType()), False)
        ])
        fvs = spark_session.createDataFrame(data, schema)
        
        # Create seeds with both positive and negative examples
        seeds = pd.DataFrame({
            '_id': [1, 2],
            'id1': [101, 102],
            'id2': [201, 202],
            'features': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 0.0]  # Mix of positive and negative
        })
        
        with patch.object(learner, '_label_everything') as mock_label_everything:
            mock_label_everything.return_value = seeds
            result = learner.train(fvs, seeds)
            
            assert mock_label_everything.called
            assert isinstance(result, pd.DataFrame)

    @patch('madmatcher_tools._internal.active_learning.ent_active_learner.SparkSession')
    @patch('madmatcher_tools._internal.active_learning.ent_active_learner.persisted')
    def test_entropy_active_learner_train_with_user_stop(self, mock_persisted, mock_spark_session, spark_session):
        """Test EntropyActiveLearner training with user stopping."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                self._trained_model = self
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                # Always use a real DataFrame if not a Spark DataFrame
                if hasattr(df, 'toPandas'):
                    result = df.toPandas()
                else:
                    result = pd.DataFrame({
                        '_id': [1, 2, 3],
                        'id1': [101, 102, 103],
                        'id2': [201, 202, 203],
                        'features': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                    })
                if not isinstance(result, pd.DataFrame):
                    result = pd.DataFrame({
                        '_id': [1, 2, 3],
                        'id1': [101, 102, 103],
                        'id2': [201, 202, 203],
                        'features': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                    })
                result[output_col] = 0.5
                return spark_session.createDataFrame(result)
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __init__(self):
                self.call_count = 0
                
            def __call__(self, id1, id2):
                self.call_count += 1
                if self.call_count == 2:  # Stop after second call
                    return -1.0
                return 1.0 if id1 == id2 else 0.0

        mock_spark_session.builder.getOrCreate.return_value = spark_session
        
        # Mock persisted context manager
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=Mock(count=Mock(return_value=100)))
        mock_context.__exit__ = Mock(return_value=None)
        mock_persisted.return_value = mock_context
        
        model = TestMLModel()
        labeler = TestLabeler()
        learner = EntropyActiveLearner(model, labeler, batch_size=1, max_iter=10)
        
        # Create test data
        data = [(1, 101, 201, [0.1, 0.2]), (2, 102, 202, [0.3, 0.4]), (3, 103, 203, [0.5, 0.6])]
        schema = StructType([
            StructField("_id", IntegerType(), False),
            StructField("id1", IntegerType(), False),
            StructField("id2", IntegerType(), False),
            StructField("features", ArrayType(DoubleType()), False)
        ])
        fvs = spark_session.createDataFrame(data, schema)
        
        # Create seeds with both positive and negative examples
        seeds = pd.DataFrame({
            '_id': [1, 2],
            'id1': [101, 102],
            'id2': [201, 202],
            'features': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 0.0]  # Mix of positive and negative
        })
        
        with patch.object(learner, '_label_everything') as mock_label_everything:
            mock_label_everything.return_value = seeds
            result = learner.train(fvs, seeds)
            
            assert isinstance(result, pd.DataFrame)
            assert labeler.call_count >= 2  # Should have called labeler at least twice

    def test_entropy_active_learner_train_with_no_positive_negative(self, spark_session):
        """Test EntropyActiveLearner with no positive or negative examples."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                return df
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0

        model = TestMLModel()
        labeler = TestLabeler()
        learner = EntropyActiveLearner(model, labeler)
        
        # Create seeds with only positive examples
        seeds = pd.DataFrame({
            '_id': [1, 2],
            'id1': [101, 102],
            'id2': [201, 202],
            'features': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 1.0]  # All positive
        })
        
        # Create test data
        data = [(1, 101, 201, [0.1, 0.2]), (2, 102, 202, [0.3, 0.4])]
        schema = StructType([
            StructField("_id", IntegerType(), False),
            StructField("id1", IntegerType(), False),
            StructField("id2", IntegerType(), False),
            StructField("features", ArrayType(DoubleType()), False)
        ])
        fvs = spark_session.createDataFrame(data, schema)
        
        with pytest.raises(RuntimeError, match="both postive and negative vectors are required for training"):
            learner.train(fvs, seeds)


@pytest.mark.unit
class TestContinuousEntropyActiveLearner:
    """Test ContinuousEntropyActiveLearner functionality."""

    def test_continuous_entropy_active_learner_creation(self, mock_labeled_data):
        """Test creating ContinuousEntropyActiveLearner."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                self._trained_model = self
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                result = df.copy()
                result[output_col] = 0.5
                return result
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0 if id1 == id2 else 0.0

        model = TestMLModel()
        labeler = TestLabeler()
        
        learner = ContinuousEntropyActiveLearner(
            model, labeler, queue_size=20, max_labeled=1000, on_demand_stop=True
        )
        
        assert learner._queue_size == 20
        assert learner._max_labeled == 1000
        assert learner._on_demand_stop is True
        assert learner._min_batch_size == 3
        assert learner._labeler == labeler
        assert isinstance(learner._model, TestMLModel)
        assert learner._model.nan_fill == model.nan_fill
        assert learner._model.use_vectors == model.use_vectors
        assert learner._model.use_floats == model.use_floats

    def test_continuous_entropy_active_learner_invalid_args(self):
        """Test ContinuousEntropyActiveLearner with invalid arguments."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                return df
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0

        model = TestMLModel()
        labeler = TestLabeler()
        
        # Test invalid queue_size
        with pytest.raises(ValueError, match="queue_size must be > 0"):
            ContinuousEntropyActiveLearner(model, labeler, queue_size=0)
        
        # Test invalid max_labeled
        with pytest.raises(ValueError, match="max_labeled must be > 0"):
            ContinuousEntropyActiveLearner(model, labeler, max_labeled=0)

    def test_continuous_entropy_active_learner_get_pos_negative(self):
        """Test _get_pos_negative method."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                return df
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0

        model = TestMLModel()
        labeler = TestLabeler()
        learner = ContinuousEntropyActiveLearner(model, labeler)
        
        # Test with mixed labels
        batch = pd.DataFrame({'label': [1.0, 0.0, 1.0, 0.0, 1.0]})
        pos, neg = learner._get_pos_negative(batch)
        assert pos == 3
        assert neg == 2

    def test_continuous_entropy_active_learner_select_training_vectors(self, spark_session):
        """Test _select_training_vectors method."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                return df
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0

        model = TestMLModel()
        labeler = TestLabeler()
        learner = ContinuousEntropyActiveLearner(model, labeler)
        
        # Create test data
        data = [(1, 101, 201, [0.1, 0.2]), (2, 102, 202, [0.3, 0.4]), (3, 103, 203, [0.5, 0.6])]
        schema = StructType([
            StructField("_id", IntegerType(), False),
            StructField("id1", IntegerType(), False),
            StructField("id2", IntegerType(), False),
            StructField("features", ArrayType(DoubleType()), False)
        ])
        fvs = spark_session.createDataFrame(data, schema)
        
        # Test selection
        ids = [1, 3]
        result = learner._select_training_vectors(fvs, ids)
        assert result.count() == 2
        result_ids = [row._id for row in result.collect()]
        assert set(result_ids) == {1, 3}

    def test_pqueue_item(self):
        """Test PQueueItem dataclass."""
        item1 = PQueueItem(0.5, {'id': 1, 'data': 'test1'})
        item2 = PQueueItem(0.3, {'id': 2, 'data': 'test2'})
        item3 = PQueueItem(0.5, {'id': 3, 'data': 'test3'})
        
        # Test ordering (lower entropy should come first)
        assert item2 < item1
        assert item2 < item3
        assert item1 == item3  # Same entropy
        
        # Test attributes
        assert item1.entropy == 0.5
        assert item1.item['id'] == 1
        assert item1.item['data'] == 'test1'

    def test_continuous_entropy_active_learner_insufficient_examples(self, spark_session):
        """Test ContinuousEntropyActiveLearner with insufficient examples."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                self._trained_model = self
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                # Return DataFrame with entropy column
                if df.count() == 0:
                    return df
                # Add entropy column to the DataFrame
                from pyspark.sql.functions import lit
                return df.withColumn(output_col, lit(0.5))
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0 if id1 == id2 else 0.0

        model = TestMLModel()
        labeler = TestLabeler()
        
        # Create a small dataset with only 2 examples (same as seeds)
        fvs_data = [
            {'_id': 1, 'id1': 101, 'id2': 201, 'features': [0.1, 0.2]},
            {'_id': 2, 'id1': 102, 'id2': 202, 'features': [0.3, 0.4]}
        ]
        fvs = spark_session.createDataFrame(fvs_data)
        
        # Create seeds with the same examples
        seeds_data = [
            {'_id': 1, 'id1': 101, 'id2': 201, 'features': [0.1, 0.2], 'label': 1.0},
            {'_id': 2, 'id1': 102, 'id2': 202, 'features': [0.3, 0.4], 'label': 0.0}
        ]
        seeds = pd.DataFrame(seeds_data)
        
        learner = ContinuousEntropyActiveLearner(model, labeler)
        
        # This should trigger the insufficient examples check and label everything
        result = learner.train(fvs, seeds)
        
        # Should return labeled data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'label' in result.columns

    def test_continuous_entropy_active_learner_empty_candidates(self, spark_session):
        """Test ContinuousEntropyActiveLearner when no candidates are left."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                self._trained_model = self
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                # Return empty DataFrame with entropy column to simulate no candidates
                if df.count() == 0:
                    return df
                # Add entropy column to the DataFrame
                from pyspark.sql.functions import lit
                return df.withColumn(output_col, lit(0.5))
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0 if id1 == id2 else 0.0

        model = TestMLModel()
        labeler = TestLabeler()
        
        # Create a dataset with more examples than seeds
        fvs_data = [
            {'_id': 1, 'id1': 101, 'id2': 201, 'features': [0.1, 0.2]},
            {'_id': 2, 'id1': 102, 'id2': 202, 'features': [0.3, 0.4]},
            {'_id': 3, 'id1': 103, 'id2': 203, 'features': [0.5, 0.6]}
        ]
        fvs = spark_session.createDataFrame(fvs_data)
        
        # Create seeds with some examples
        seeds_data = [
            {'_id': 1, 'id1': 101, 'id2': 201, 'features': [0.1, 0.2], 'label': 1.0},
            {'_id': 2, 'id1': 102, 'id2': 202, 'features': [0.3, 0.4], 'label': 0.0}
        ]
        seeds = pd.DataFrame(seeds_data)
        
        learner = ContinuousEntropyActiveLearner(model, labeler, queue_size=5)
        
        # This should handle the empty candidates gracefully
        result = learner.train(fvs, seeds)
        
        # Should return labeled data
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 2  # At least the seeds


@pytest.mark.unit
class TestActiveLearningIntegration:
    """Test integration scenarios for active learning."""

    def test_entropy_active_learner_train_with_no_positive_negative(self, spark_session):
        """Test EntropyActiveLearner with no positive or negative examples."""
        class TestMLModel(MLModel):
            def __init__(self):
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
                return self
            
            def predict(self, df, vector_col, output_col):
                return df
            
            def prediction_conf(self, df, vector_col, label_column):
                return df
            
            def entropy(self, df, vector_col, output_col):
                return df
            
            def params_dict(self):
                return {}
            
            def prep_fvs(self, fvs):
                return fvs

        class TestLabeler(Labeler):
            def __call__(self, id1, id2):
                return 1.0

        model = TestMLModel()
        labeler = TestLabeler()
        learner = EntropyActiveLearner(model, labeler)
        
        # Create seeds with only positive examples
        seeds = pd.DataFrame({
            '_id': [1, 2],
            'id1': [101, 102],
            'id2': [201, 202],
            'features': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 1.0]  # All positive
        })
        
        # Create test data
        data = [(1, 101, 201, [0.1, 0.2]), (2, 102, 202, [0.3, 0.4])]
        schema = StructType([
            StructField("_id", IntegerType(), False),
            StructField("id1", IntegerType(), False),
            StructField("id2", IntegerType(), False),
            StructField("features", ArrayType(DoubleType()), False)
        ])
        fvs = spark_session.createDataFrame(data, schema)
        
        with pytest.raises(RuntimeError, match="both postive and negative vectors are required for training"):
            learner.train(fvs, seeds)
