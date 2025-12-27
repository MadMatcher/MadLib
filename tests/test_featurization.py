"""Tests for MadLib._internal.featurization module.

This module provides tests for feature creation and featurization functions.
"""
import pandas as pd
import numpy as np

from MadLib._internal import featurization
from MadLib._internal.feature.feature import ExactMatchFeature, RelDiffFeature
from MadLib._internal.feature.token_feature import JaccardFeature
from MadLib._internal.tokenizer.tokenizer import StrippedWhiteSpaceTokenizer


class TestGetFunctions:
    """Tests for getter functions."""

    def test_get_base_sim_functions(self):
        """Test get_base_sim_functions returns expected list."""
        functions = featurization.get_base_sim_functions()
        
        assert isinstance(functions, list)
        assert len(functions) > 0
        assert all(callable(f) or isinstance(f, type) for f in functions)

    def test_get_base_tokenizers(self):
        """Test get_base_tokenizers returns expected list."""
        tokenizers = featurization.get_base_tokenizers()
        
        assert isinstance(tokenizers, list)
        assert len(tokenizers) > 0
        assert all(hasattr(t, 'tokenize') for t in tokenizers)

    def test_get_extra_tokenizers(self):
        """Test get_extra_tokenizers returns expected list."""
        tokenizers = featurization.get_extra_tokenizers()
        
        assert isinstance(tokenizers, list)
        assert len(tokenizers) > 0
        assert all(hasattr(t, 'tokenize') for t in tokenizers)


class TestBuildCache:
    """Tests for BuildCache class."""

    def test_init(self):
        """Test BuildCache initialization."""
        cache = featurization.BuildCache()
        assert cache._cache == []
        assert cache._lock is not None

    def test_add_or_get_new(self):
        """Test add_or_get adds new builder."""
        cache = featurization.BuildCache()
        builder = object()
        
        result = cache.add_or_get(builder)
        
        assert result is builder
        assert len(cache._cache) == 1
        assert cache._cache[0] is builder

    def test_add_or_get_existing(self):
        """Test add_or_get returns existing builder."""
        cache = featurization.BuildCache()
        builder = object()
        
        result1 = cache.add_or_get(builder)
        result2 = cache.add_or_get(builder)
        
        assert result1 is builder
        assert result2 is builder
        assert len(cache._cache) == 1

    def test_clear(self):
        """Test clear removes all builders."""
        cache = featurization.BuildCache()
        cache.add_or_get(object())
        cache.add_or_get(object())
        
        cache.clear()
        
        assert len(cache._cache) == 0


class TestCreateFeatures:
    """Tests for create_features function."""

    def test_create_features_pandas(self, a_df, b_df):
        """Test create_features with pandas DataFrames."""
        a_pdf = a_df.toPandas()
        b_pdf = b_df.toPandas()

        features = featurization.create_features(
            a_pdf, b_pdf, ['a_attr', 'a_num'], ['a_attr', 'a_num']
        )

        assert isinstance(features, list)
        assert len(features) > 0
        feature_strs = [str(f) for f in features]
        assert any('a_attr' in s for s in feature_strs)
        assert any(isinstance(f, RelDiffFeature) for f in features)

    def test_create_features_spark(self, spark_session, a_df, b_df):
        """Test create_features with Spark DataFrames."""
        features = featurization.create_features(
            a_df, b_df, ['a_attr', 'a_num'], ['a_attr', 'a_num']
        )

        assert isinstance(features, list)
        assert len(features) > 0
        feature_strs = [str(f) for f in features]
        assert any('a_attr' in s for s in feature_strs)
        assert any(isinstance(f, RelDiffFeature) for f in features)

    def test_create_features_custom_sim_functions(self, a_df, b_df):
        """Test create_features with custom similarity functions."""
        custom_sim = [JaccardFeature]

        features = featurization.create_features(
            a_df, b_df, ['a_attr'], ['a_attr'],
            sim_functions=custom_sim
        )

        assert isinstance(features, list)
        assert len(features) > 0
        feature_strs = [str(f).lower() for f in features]
        assert any('exact_match' in s for s in feature_strs)

    def test_create_features_custom_tokenizers(self, a_df, b_df):
        """Test create_features with custom tokenizers."""
        custom_tokenizer = [StrippedWhiteSpaceTokenizer()]

        features = featurization.create_features(
            a_df, b_df, ['a_attr'], ['a_attr'],
            tokenizers=custom_tokenizer
        )

        assert isinstance(features, list)
        assert len(features) > 0

    def test_create_features_null_threshold(self, spark_session):
        """Test create_features drops columns with high null percentage."""
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

        a_schema = StructType([
            StructField('_id', IntegerType(), nullable=False),
            StructField('a_attr', StringType(), nullable=True),
            StructField('a_null', DoubleType(), nullable=True),
        ])
        b_schema = StructType([
            StructField('_id', IntegerType(), nullable=False),
            StructField('a_attr', StringType(), nullable=True),
            StructField('a_null', DoubleType(), nullable=True),
        ])

        a_df = spark_session.createDataFrame([
            {'_id': 10, 'a_attr': 'a', 'a_null': None},
            {'_id': 11, 'a_attr': 'b', 'a_null': None},
            {'_id': 12, 'a_attr': 'c', 'a_null': None},
        ], schema=a_schema)
        b_df = spark_session.createDataFrame([
            {'_id': 20, 'a_attr': 'a', 'a_null': None},
        ], schema=b_schema)
        features = featurization.create_features(
            a_df, b_df, ['a_attr', 'a_null'], ['a_attr', 'a_null'],
            null_threshold=0.5
        )

        feature_strs = [str(f) for f in features]
        assert not any('a_null' in s or 'b_null' in s for s in feature_strs)

    def test_create_features_single_table(self, spark_session):
        """Test create_features with B=None (single table)."""
        a_df = spark_session.createDataFrame([
            {'_id': 10, 'a_attr': 'a', 'a_num': 1.0},
            {'_id': 11, 'a_attr': 'b', 'a_num': 2.0},
        ])
        
        features = featurization.create_features(
            a_df, None, ['a_attr', 'a_num'], []
        )
        
        assert isinstance(features, list)
        assert len(features) > 0


class TestFeaturize:
    """Tests for featurize function."""

    def test_featurize_pandas(self, a_df, b_df):
        """Test featurize with pandas DataFrames."""
        a_pdf = a_df.toPandas()
        b_pdf = b_df.toPandas()

        features = featurization.create_features(
            a_pdf, b_pdf, ['a_attr', 'a_num'], ['a_attr', 'a_num']
        )

        candidates = pd.DataFrame({
            'id1_list': [[10], [11], [12]],
            'id2': [20, 21, 22]
        })

        fvs = featurization.featurize(
            features, a_pdf, b_pdf, candidates
        )

        assert isinstance(fvs, pd.DataFrame)
        assert 'feature_vectors' in fvs.columns
        assert 'id1' in fvs.columns
        assert 'id2' in fvs.columns
        assert len(fvs) == 3
        assert isinstance(fvs['feature_vectors'].iloc[0], (list, np.ndarray))

    def test_featurize_spark(self, spark_session, a_df, b_df):
        """Test featurize with Spark DataFrames."""
        features = featurization.create_features(
            a_df, b_df, ['a_attr', 'a_num'], ['a_attr', 'a_num']
        )

        candidates = spark_session.createDataFrame([
            {'id1_list': [10], 'id2': 20},
            {'id1_list': [11], 'id2': 21},
            {'id1_list': [12], 'id2': 22}
        ])

        fvs = featurization.featurize(
            features, a_df, b_df, candidates
        )

        assert hasattr(fvs, 'count')
        assert fvs.count() == 3
        assert 'feature_vectors' in [col.name for col in fvs.schema.fields]
        assert 'id1' in [col.name for col in fvs.schema.fields]
        assert 'id2' in [col.name for col in fvs.schema.fields]

    def test_featurize_custom_output_col(self, a_df, b_df):
        """Test featurize with custom output column name."""
        a_pdf = a_df.toPandas()
        b_pdf = b_df.toPandas()

        features = featurization.create_features(
            a_pdf, b_pdf, ['a_attr'], ['a_attr']
        )

        candidates = pd.DataFrame({
            'id1_list': [[10]],
            'id2': [20]
        })

        fvs = featurization.featurize(
            features, a_pdf, b_pdf, candidates,
            output_col='custom_fv'
        )

        assert 'custom_fv' in fvs.columns
        assert 'feature_vectors' not in fvs.columns

    def test_featurize_custom_fill_na(self, a_df, b_df):
        """Test featurize with custom fill_na value."""
        a_pdf = a_df.toPandas()
        b_pdf = b_df.toPandas()

        features = featurization.create_features(
            a_pdf, b_pdf, ['a_attr'], ['a_attr']
        )

        candidates = pd.DataFrame({
            'id1_list': [[10]],
            'id2': [20]
        })

        fvs = featurization.featurize(
            features, a_pdf, b_pdf, candidates,
            fill_na=-1.0
        )

        assert isinstance(fvs, pd.DataFrame)
        assert len(fvs) > 0

    def test_featurize_has_score_column(self, a_df, b_df):
        """Test featurize creates score column."""
        a_pdf = a_df.toPandas()
        b_pdf = b_df.toPandas()

        features = featurization.create_features(
            a_pdf, b_pdf, ['a_attr'], ['a_attr']
        )

        candidates = pd.DataFrame({
            'id1_list': [[10]],
            'id2': [20]
        })

        fvs = featurization.featurize(
            features, a_pdf, b_pdf, candidates
        )

        assert 'score' in fvs.columns
        assert fvs['score'].dtype in [np.float64, np.float32]


class TestScore:
    """Tests for score function."""

    def test_score_exists(self):
        """Test that score function exists (currently not implemented)."""
        assert callable(featurization.score)
        
        fvs = pd.DataFrame({
            'id1': [10],
            'id2': [20],
            'feature_vectors': [[0.1, 0.2, 0.3]]
        })
        features = pd.DataFrame({
            'feature_name': ['exact_match']
        })
        
        result = featurization.score(fvs, features)
        assert result is None


class TestInternalHelpers:
    """Tests for internal helper functions."""

    def test_get_pos_cor_features(self):
        """Test _get_pos_cor_features identifies positive correlated."""
        features = [
            ExactMatchFeature('a_attr', 'a_attr'),
            RelDiffFeature('a_num', 'a_num'),
            JaccardFeature(
                'a_attr', 'a_attr', StrippedWhiteSpaceTokenizer()
            )
        ]

        pos_cor = featurization._get_pos_cor_features(features)

        assert isinstance(pos_cor, list)
        assert len(pos_cor) == len(features)
        assert pos_cor[0] == 1
        assert pos_cor[2] == 1

    def test_score_fvs_adds_score_column(self, spark_session):
        """Test _score_fvs adds score column."""
        fvs = spark_session.createDataFrame([
            {
                '_id': 1, 'id1': 10, 'id2': 20,
                'feature_vectors': [1.0, 0.5, 0.8]
            },
            {
                '_id': 2, 'id1': 11, 'id2': 21,
                'feature_vectors': [0.0, 0.3, 0.2]
            },
        ])

        pos_cor = [1, 1, 1]

        scored = featurization._score_fvs(fvs, 'feature_vectors', pos_cor)

        assert 'score' in [col.name for col in scored.schema.fields]
        scores = scored.select('score').collect()
        assert len(scores) == 2
        assert scores[0]['score'] > 0
        assert scores[1]['score'] > 0
