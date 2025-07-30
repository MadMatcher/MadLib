import pytest
import pandas as pd
import numpy as np
from MadLib._internal.featurization import create_features
from MadLib._internal.feature.feature import ExactMatchFeature, RelDiffFeature

class DummyTokenizer:
    NAME = 'dummy_tokenizer'
    def out_col_name(self, base):
        return f'dummy({base})'
    def tokenize(self, x):
        return [str(x)] if x is not None else []
    def tokenize_set(self, x):
        return set(self.tokenize(x))
    def __str__(self):
        return self.NAME

@pytest.fixture
def df_all_null():
    return pd.DataFrame({'a': [None, None], 'b': [None, None]})

@pytest.fixture
def df_numeric():
    return pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})

@pytest.fixture
def df_mixed():
    return pd.DataFrame({'a': [1, None, 3], 'b': ['x', 'y', None]})

def test_create_features_empty():
    df = pd.DataFrame()
    feats = create_features(df, df, [], [])
    assert feats == []

def test_create_features_all_null(df_all_null):
    feats = create_features(df_all_null, df_all_null, ['a', 'b'], ['a', 'b'])
    # Should drop all columns
    assert all(isinstance(f, ExactMatchFeature) for f in feats) or feats == []

def test_create_features_numeric(df_numeric):
    feats = create_features(df_numeric, df_numeric, ['a', 'b'], ['a', 'b'])
    # Should include RelDiffFeature for numeric columns
    assert any(isinstance(f, RelDiffFeature) for f in feats)

def test_create_features_tokenizer(df_mixed):
    # Use data with longer strings to increase token count
    df = pd.DataFrame({'a': ['one two three', 'four five six', 'seven eight nine'],
                       'b': ['alpha beta gamma', 'delta epsilon zeta', 'eta theta iota']})
    feats = create_features(df, df, ['a', 'b'], ['a', 'b'], tokenizers=[DummyTokenizer()])
    # Should create at least one feature
    assert len(feats) > 0

def test_create_features_null_threshold(df_numeric):
    # Set null_threshold to 0 to drop all columns
    feats = create_features(df_numeric, df_numeric, ['a', 'b'], ['a', 'b'], null_threshold=0)
    assert feats == []

def test_create_features_only_strings_with_tokenizer():
    df = pd.DataFrame({'a': ['foo bar', 'baz qux', 'quux corge'],
                       'b': ['alpha beta', 'gamma delta', 'epsilon zeta']})
    feats = create_features(df, df, ['a', 'b'], ['a', 'b'], tokenizers=[DummyTokenizer()])
    # Should create at least one feature
    assert len(feats) > 0 