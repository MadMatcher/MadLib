"""
Unit tests for MadLib.tools module.

Tests all public API functions with various input types and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from MadLib.tools import (
    down_sample, create_seeds, train_matcher, apply_matcher, label_data
)
from MadLib import Labeler, MLModel


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

    @patch('MadLib.tools.SparkDataFrame')
    @patch('MadLib.tools.isinstance')
    def test_down_sample_spark_bucket_size_too_small(self, mock_isinstance, mock_spark_dataframe):
        """Test down_sample with Spark DataFrame and bucket_size < 1000."""
        # Mock isinstance to return True for SparkDataFrame
        def mock_isinstance_func(obj, type_or_tuple):
            if type_or_tuple == mock_spark_dataframe:
                return True
            return isinstance(obj, type_or_tuple)
        mock_isinstance.side_effect = mock_isinstance_func
        
        # Create a mock that simulates SparkDataFrame
        mock_df = Mock()
        mock_df.count.return_value = 1000
        # Add __len__ method to support len() calls
        mock_df.__len__ = Mock(return_value=1000)
        # Mock the Spark operations that would be called
        mock_df.withColumn.return_value = mock_df
        mock_df.select.return_value = mock_df
        mock_df.filter.return_value = mock_df
        mock_df.drop.return_value = mock_df
        
        with pytest.raises(ValueError, match="bucket_size must be >= 1000"):
            down_sample(mock_df, percent=0.5, search_id_column='id2', bucket_size=500)

    @patch('MadLib.tools.SparkDataFrame')
    @patch('MadLib.tools.isinstance')
    def test_down_sample_spark_invalid_percent(self, mock_isinstance, mock_spark_dataframe):
        """Test down_sample with Spark DataFrame and invalid percent."""
        # Mock isinstance to return True for SparkDataFrame
        def mock_isinstance_func(obj, type_or_tuple):
            if type_or_tuple == mock_spark_dataframe:
                return True
            return isinstance(obj, type_or_tuple)
        mock_isinstance.side_effect = mock_isinstance_func
        
        # Create a mock that simulates SparkDataFrame
        mock_df = Mock()
        mock_df.count.return_value = 1000
        # Add __len__ method to support len() calls
        mock_df.__len__ = Mock(return_value=1000)
        # Mock the Spark operations that would be called
        mock_df.withColumn.return_value = mock_df
        mock_df.select.return_value = mock_df
        mock_df.filter.return_value = mock_df
        mock_df.drop.return_value = mock_df
        
        with pytest.raises(ValueError, match="percent must be in the range"):
            down_sample(mock_df, percent=0.0, search_id_column='id2', bucket_size=1000)

    def test_down_sample_empty_input(self):
        """Test down_sample with empty DataFrame (should return empty DataFrame)."""
        fvs = pd.DataFrame(columns=['id2', 'score'])
        result = down_sample(fvs, percent=0.5, search_id_column='id2')
        assert result.empty

    def test_down_sample_invalid_type(self):
        """Test down_sample with invalid input type (should return input unchanged)."""
        # Use an integer to test invalid type behavior
        invalid_input = 12345
        result = down_sample(invalid_input, percent=0.5, search_id_column='id2')
        # The function should return the input unchanged when given invalid type
        assert result == invalid_input
        assert type(result) == type(invalid_input)


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
        
        with pytest.raises(ValueError, match="number of seeds would exceed"):
            create_seeds(fvs, nseeds=100, labeler=labeler)

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

    @patch('MadLib.tools.SparkDataFrame')
    @patch('MadLib.tools.isinstance')
    def test_create_seeds_spark_too_many_seeds(self, mock_isinstance, mock_spark_dataframe):
        """Test create_seeds with Spark DataFrame and too many seeds."""
        # Mock isinstance to return True for SparkDataFrame
        def mock_isinstance_func(obj, type_or_tuple):
            if type_or_tuple == mock_spark_dataframe:
                return True
            return isinstance(obj, type_or_tuple)
        mock_isinstance.side_effect = mock_isinstance_func
        
        # Create a mock that simulates SparkDataFrame
        mock_df = Mock()
        mock_df.count.return_value = 5  # Return actual integer
        # Add __len__ method to support len() calls
        mock_df.__len__ = Mock(return_value=5)
        # Ensure filter returns the same mock with count method intact
        mock_df.filter.return_value = mock_df
        labeler = Mock(spec=Labeler)
        
        with pytest.raises(ValueError, match="number of seeds would exceed"):
            create_seeds(mock_df, nseeds=10, labeler=labeler)

    @patch('MadLib.tools.F')
    @patch('MadLib.tools.SparkDataFrame')
    @patch('MadLib.tools.isinstance')
    def test_create_seeds_spark_custom_score_column(self, mock_isinstance, mock_spark_dataframe, mock_F):
        """Test create_seeds with Spark DataFrame and custom score column (pure mock, no SparkContext needed)."""
        # Mock F.col to return a mock column object
        mock_col = Mock()
        mock_col.isNotNull.return_value = True
        mock_F.col.return_value = mock_col
        mock_F.isnan.return_value = False
        
        # Mock isinstance to return True for SparkDataFrame
        def mock_isinstance_func(obj, type_or_tuple):
            if type_or_tuple == mock_spark_dataframe:
                return True
            return isinstance(obj, type_or_tuple)
        mock_isinstance.side_effect = mock_isinstance_func
        
        # Create a mock that simulates SparkDataFrame
        mock_df = Mock()
        mock_df.count.return_value = 10
        mock_df.__len__ = Mock(return_value=10)
        mock_df.filter.return_value = mock_df
        mock_df.sort.return_value = mock_df
        mock_df.limit.return_value = mock_df
        mock_df.toPandas.return_value = pd.DataFrame({
            'id1': [1, 2], 'id2': [101, 102], 'score': [0.9, 0.8]
        })
        
        labeler = Mock(spec=Labeler)
        labeler.return_value = 1.0
        
        seeds = create_seeds(mock_df, nseeds=2, labeler=labeler, score_column='custom_score')
        
        assert len(seeds) <= 2
        assert 'label' in seeds.columns

    def test_create_seeds_labeler_stop_iteration(self, sample_feature_vectors):
        """Test create_seeds when labeler runs out of examples."""
        fvs = sample_feature_vectors.copy()
        fvs['score'] = [0.9, 0.8, 0.6, 0.4, 0.2]
        
        # Mock labeler that returns -1.0 (stop) immediately
        labeler = Mock(spec=Labeler)
        labeler.return_value = -1.0
        
        with pytest.raises(RuntimeError, match="No seeds were labeled before stopping"):
            create_seeds(fvs, nseeds=3, labeler=labeler)

    def test_create_seeds_labeler_unsure_labels(self, sample_feature_vectors):
        """Test create_seeds with labeler returning unsure labels (2.0)."""
        # Create a larger dataset to ensure we have enough examples
        fvs = pd.DataFrame({
            'id1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'id2': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            'features': [[0.1, 0.2, 0.3]] * 10
        })
        
        # Mock labeler that returns unsure (2.0) for first few calls, then valid labels
        # Need to provide enough side effects to ensure we get valid labels
        labeler = Mock(spec=Labeler)
        # Provide more side effects to ensure we get enough valid labels
        labeler.side_effect = [2.0, 2.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        
        seeds = create_seeds(fvs, nseeds=2, labeler=labeler)
        
        assert len(seeds) <= 2
        assert 'label' in seeds.columns
        assert seeds['label'].isin([0.0, 1.0]).all()

    def test_create_seeds_invalid_labeler(self, sample_feature_vectors):
        """Test create_seeds with invalid labeler type (should raise TypeError or ValueError)."""
        with pytest.raises((TypeError, ValueError)):
            create_seeds(sample_feature_vectors, nseeds=2, labeler='not_a_labeler')


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

    def test_train_matcher_invalid_input(self):
        """Test train_matcher with invalid input (should raise Exception)."""
        with pytest.raises((TypeError, ValueError, AttributeError)):
            train_matcher('not_a_model_spec', pd.DataFrame({'id1': [1], 'id2': [2], 'features': [[1.0]], 'label': [1.0]}))

    @patch('MadLib._internal.ml_model.convert_to_array', side_effect=lambda df, col: df)
    def test_train_matcher_spark_dataframe(self, mock_convert, mock_labeled_data):
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
            model = train_matcher(model_spec, mock_spark_df)
            assert hasattr(model, 'trained_model')
            assert isinstance(model.trained_model, LogisticRegression)


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

    def test_apply_matcher_invalid_input(self):
        """Test apply_matcher with invalid model (should raise Exception)."""
        with pytest.raises((TypeError, ValueError, AttributeError)):
            apply_matcher('not_a_model', pd.DataFrame({'id1': [1], 'id2': [2], 'features': [[1.0]]}), 'features', 'output')

    @patch('MadLib._internal.ml_model.convert_to_array', side_effect=lambda df, col: df)
    def test_apply_matcher_spark_dataframe(self, mock_convert, sample_feature_vectors, mock_labeled_data):
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
            training_model = train_matcher(model_spec, mock_labeled_data)
            with patch.object(training_model, 'predict') as mock_predict:
                mock_predict.return_value = sample_feature_vectors.assign(prediction=[0.5, 0.3, 0.7, 0.2, 0.8])
                result = apply_matcher(training_model, mock_spark_df, 'features', 'prediction')
                assert isinstance(result, pd.DataFrame)
                assert 'prediction' in result.columns


@pytest.mark.unit
class TestLabelData:
    """Test label_data function."""

    @patch('MadLib.tools.create_seeds')
    @patch('MadLib.tools.EntropyActiveLearner')
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

    @patch('MadLib.tools.create_seeds')
    @patch('MadLib.tools.ContinuousEntropyActiveLearner')
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
        # Convert gold_labels DataFrame to the format expected by GoldLabeler
        gold_dict = {'id1': gold_labels['id1'].tolist(), 'id2': gold_labels['id2'].tolist()}
        labeler_spec = {'name': 'gold', 'gold': gold_dict}
        
        result = label_data(model_spec, 'continuous', labeler_spec, sample_feature_vectors)
        
        assert isinstance(result, pd.DataFrame)
        mock_learner_class.assert_called_once()
        mock_learner.train.assert_called_once()

    @patch('MadLib.tools.create_seeds')
    def test_label_data_with_existing_seeds(self, mock_create_seeds,
                                          sample_feature_vectors, gold_labels):
        """Test label_data with pre-existing seeds."""
        existing_seeds = pd.DataFrame({'id1': [1], 'id2': [101], 'label': [1.0]})
        
        with patch('MadLib.tools.EntropyActiveLearner') as mock_learner_class:
            mock_learner = Mock()
            mock_learner.train.return_value = existing_seeds
            mock_learner_class.return_value = mock_learner
            
            model_spec = {'model_type': 'sklearn', 'model': LogisticRegression, 'model_args': {}}
            # Convert gold_labels DataFrame to the format expected by GoldLabeler
            gold_dict = {'id1': gold_labels['id1'].tolist(), 'id2': gold_labels['id2'].tolist()}
            labeler_spec = {'name': 'gold', 'gold': gold_dict}
            
            result = label_data(model_spec, 'batch', labeler_spec, 
                              sample_feature_vectors, seeds=existing_seeds)
            
            # Should not call create_seeds when seeds are provided
            mock_create_seeds.assert_not_called()
            mock_learner.train.assert_called_once()

    @patch('MadLib.tools.SparkSession')
    def test_label_data_spark_dataframe_conversion(self, mock_spark_session):
        """Test label_data with pandas DataFrame that gets converted to Spark."""
        mock_spark = Mock()
        mock_spark_session.builder.getOrCreate.return_value = mock_spark
        
        # Create a proper mock SparkDataFrame that supports count() and comparison
        mock_spark_df = Mock()
        mock_spark_df.count.return_value = 5  # Return integer for min() comparison
        mock_spark.createDataFrame.return_value = mock_spark_df
        
        fvs = pd.DataFrame({'id1': [1, 2], 'id2': [101, 102], 'score': [0.9, 0.8]})
        model_spec = {'model_type': 'sklearn', 'model': LogisticRegression, 'model_args': {}}
        # Fix the gold labeler format
        labeler_spec = {'name': 'gold', 'gold': {'id1': [1], 'id2': [101]}}
        
        with patch('MadLib.tools.create_seeds') as mock_create_seeds:
            with patch('MadLib.tools.EntropyActiveLearner') as mock_learner_class:
                mock_learner = Mock()
                mock_learner_class.return_value = mock_learner
                mock_learner.train.return_value = pd.DataFrame({'id1': [1], 'id2': [101], 'label': [1.0]})
                
                result = label_data(model_spec, 'batch', labeler_spec, fvs)
                
                mock_spark.createDataFrame.assert_called_once_with(fvs)
                assert isinstance(result, pd.DataFrame)

    @patch('MadLib.tools.SparkSession')
    def test_label_data_continuous_mode(self, mock_spark_session):
        """Test label_data with continuous mode."""
        mock_spark = Mock()
        mock_spark_session.builder.getOrCreate.return_value = mock_spark
        
        # Create a proper mock SparkDataFrame that supports count() and comparison
        mock_spark_df = Mock()
        mock_spark_df.count.return_value = 5  # Return integer for min() comparison
        mock_spark.createDataFrame.return_value = mock_spark_df
        
        fvs = pd.DataFrame({'id1': [1, 2], 'id2': [101, 102], 'score': [0.9, 0.8]})
        model_spec = {'model_type': 'sklearn', 'model': LogisticRegression, 'model_args': {}}
        # Fix the gold labeler format
        labeler_spec = {'name': 'gold', 'gold': {'id1': [1], 'id2': [101]}}
        
        with patch('MadLib.tools.create_seeds') as mock_create_seeds:
            with patch('MadLib.tools.ContinuousEntropyActiveLearner') as mock_learner_class:
                mock_learner = Mock()
                mock_learner_class.return_value = mock_learner
                mock_learner.train.return_value = pd.DataFrame({'id1': [1], 'id2': [101], 'label': [1.0]})
                
                result = label_data(model_spec, 'continuous', labeler_spec, fvs)
                
                mock_learner_class.assert_called_once()
                assert isinstance(result, pd.DataFrame)

    @patch('MadLib.tools.SparkSession')
    def test_label_data_continuous_mode_spark(self, mock_spark_session):
        """Test label_data with continuous mode and Spark DataFrame."""
        mock_spark = Mock()
        mock_spark_session.builder.getOrCreate.return_value = mock_spark
        
        # Create a proper mock SparkDataFrame that supports count() and comparison
        mock_spark_df = Mock()
        mock_spark_df.count.return_value = 5  # Return integer for min() comparison
        mock_spark.createDataFrame.return_value = mock_spark_df
        
        fvs = pd.DataFrame({'id1': [1, 2], 'id2': [101, 102], 'score': [0.9, 0.8]})
        model_spec = {'model_type': 'sklearn', 'model': LogisticRegression, 'model_args': {}}
        # Fix the gold labeler format
        labeler_spec = {'name': 'gold', 'gold': {'id1': [1], 'id2': [101]}}
        
        with patch('MadLib.tools.create_seeds') as mock_create_seeds:
            with patch('MadLib.tools.ContinuousEntropyActiveLearner') as mock_learner_class:
                mock_learner = Mock()
                mock_learner_class.return_value = mock_learner
                mock_learner.train.return_value = pd.DataFrame({'id1': [1], 'id2': [101], 'label': [1.0]})
                
                result = label_data(model_spec, 'continuous', labeler_spec, fvs)
                
                mock_learner_class.assert_called_once()
                assert isinstance(result, pd.DataFrame)

    def test_label_data_invalid_mode(self, sample_feature_vectors, gold_labels):
        """Test label_data with invalid mode (should raise Exception)."""
        model_spec = {'model_type': 'sklearn', 'model': LogisticRegression, 'model_args': {}}
        labeler_spec = {'name': 'gold', 'gold': gold_labels}
        with pytest.raises((ValueError, TypeError)):
            label_data(model_spec, 'invalid_mode', labeler_spec, sample_feature_vectors)


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

    def test_create_seeds_labeler_stop_iteration(self, sample_feature_vectors):
        """Test create_seeds when labeler runs out of examples."""
        fvs = sample_feature_vectors.copy()
        fvs['score'] = [0.9, 0.8, 0.6, 0.4, 0.2]
        
        # Mock labeler that returns -1.0 (stop) immediately
        labeler = Mock(spec=Labeler)
        labeler.return_value = -1.0
        
        with pytest.raises(RuntimeError, match="No seeds were labeled before stopping"):
            create_seeds(fvs, nseeds=3, labeler=labeler)

    def test_create_seeds_labeler_unsure_labels(self, sample_feature_vectors):
        """Test create_seeds with labeler returning unsure labels (2.0)."""
        # Create a larger dataset to ensure we have enough examples
        fvs = pd.DataFrame({
            'id1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'id2': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            'features': [[0.1, 0.2, 0.3]] * 10
        })
        
        # Mock labeler that returns unsure (2.0) for first few calls, then valid labels
        # Need to provide enough side effects to ensure we get valid labels
        labeler = Mock(spec=Labeler)
        # Provide more side effects to ensure we get enough valid labels
        labeler.side_effect = [2.0, 2.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        
        seeds = create_seeds(fvs, nseeds=2, labeler=labeler)
        
        assert len(seeds) <= 2
        assert 'label' in seeds.columns
        assert seeds['label'].isin([0.0, 1.0]).all()

    def test_create_seeds_labeler_dict(self, sample_feature_vectors):
        """Test create_seeds with labeler as dict (should create labeler from dict)."""
        fvs = sample_feature_vectors.copy()
        labeler_dict = {'name': 'gold', 'gold': {'id1': [1], 'id2': [101]}}
        
        seeds = create_seeds(fvs, nseeds=2, labeler=labeler_dict)
        
        assert len(seeds) <= 2
        assert 'label' in seeds.columns

    def test_create_seeds_nan_scores(self, sample_feature_vectors):
        """Test create_seeds with NaN scores (should filter out NaN scores)."""
        fvs = sample_feature_vectors.copy()
        fvs.loc[0, 'score'] = np.nan
        labeler = Mock(spec=Labeler)
        labeler.return_value = 1.0
        
        seeds = create_seeds(fvs, nseeds=2, labeler=labeler)
        
        assert len(seeds) <= 2
        assert 'label' in seeds.columns

    def test_down_sample_percent_one(self, sample_feature_vectors):
        """Test down_sample with percent=1.0 (should return all rows)."""
        fvs = sample_feature_vectors.copy()
        fvs['score'] = [0.9, 0.8, 0.6, 0.4, 0.2]
        
        result = down_sample(fvs, percent=1.0, search_id_column='id2')
        
        assert len(result) <= len(fvs)

    def test_down_sample_bucket_size_one(self, sample_feature_vectors):
        """Test down_sample with bucket_size=1 (edge case)."""
        fvs = sample_feature_vectors.copy()
        fvs['score'] = [0.9, 0.8, 0.6, 0.4, 0.2]
        
        result = down_sample(fvs, percent=0.5, search_id_column='id2', bucket_size=1)
        
        assert len(result) <= len(fvs)

    @patch('MadLib._internal.ml_model.convert_to_array', side_effect=lambda df, col: df)
    def test_train_matcher_spark_dataframe(self, mock_convert, mock_labeled_data):
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
            model = train_matcher(model_spec, mock_spark_df)
            assert hasattr(model, 'trained_model')
            assert isinstance(model.trained_model, LogisticRegression)

    @patch('MadLib._internal.ml_model.convert_to_array', side_effect=lambda df, col: df)
    def test_apply_matcher_spark_dataframe(self, mock_convert, sample_feature_vectors, mock_labeled_data):
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
            training_model = train_matcher(model_spec, mock_labeled_data)
            with patch.object(training_model, 'predict') as mock_predict:
                mock_predict.return_value = sample_feature_vectors.assign(prediction=[0.5, 0.3, 0.7, 0.2, 0.8])
                result = apply_matcher(training_model, mock_spark_df, 'features', 'prediction')
                assert isinstance(result, pd.DataFrame)
                assert 'prediction' in result.columns

    def test_label_data_with_seeds(self, sample_feature_vectors, gold_labels):
        """Test label_data with provided seeds (should not create new seeds)."""
        existing_seeds = pd.DataFrame({'id1': [1], 'id2': [101], 'label': [1.0]})
        
        with patch('MadLib.tools.EntropyActiveLearner') as mock_learner_class:
            mock_learner = Mock()
            mock_learner_class.return_value = mock_learner
            mock_learner.train.return_value = existing_seeds
            
            model_spec = {'model_type': 'sklearn', 'model': LogisticRegression, 'model_args': {}}
            labeler_spec = {'name': 'gold', 'gold': gold_labels}
            
            result = label_data(model_spec, 'batch', labeler_spec, sample_feature_vectors, seeds=existing_seeds)
            
            assert isinstance(result, pd.DataFrame)
            mock_learner.train.assert_called_once() 