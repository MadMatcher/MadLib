"""
Tests for token_feature module.

This module tests the TokenFeature base class and its implementations including
JaccardFeature, OverlapCoeffFeature, CosineFeature, and MongeElkanFeature.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from MadLib._internal.feature.token_feature import (
    TokenFeature, JaccardFeature, OverlapCoeffFeature, CosineFeature, MongeElkanFeature,
    _overlap, _jaccard, _overlap_coeff
)
from MadLib._internal.tokenizer.tokenizer import Tokenizer


class MockTokenizer(Tokenizer):
    """Mock tokenizer for testing."""
    
    NAME = "mock_tokenizer"
    
    def tokenize(self, text):
        if text is None:
            return None
        return text.lower().split()
    
    def tokenize_set(self, text):
        if text is None:
            return None
        return set(text.lower().split())
    
    def tokenize_spark(self, col):
        return col
    
    def out_col_name(self, base):
        return f"tokens({base})"


@pytest.mark.unit
class TestTokenFeatureBase:
    """Test TokenFeature base class."""

    def test_token_feature_creation(self):
        """Test creating TokenFeature."""
        tokenizer = MockTokenizer()
        
        # This should raise an error since TokenFeature is abstract
        with pytest.raises(TypeError):
            TokenFeature("attr1", "attr2", tokenizer)

    def test_token_feature_abstract_methods(self):
        """Test that TokenFeature requires sim_func implementation."""
        tokenizer = MockTokenizer()
        
        class ConcreteTokenFeature(TokenFeature):
            pass  # Missing sim_func implementation
        
        with pytest.raises(TypeError):
            ConcreteTokenFeature("attr1", "attr2", tokenizer)

    def test_token_feature_with_concrete_implementation(self):
        """Test TokenFeature with concrete sim_func implementation."""
        tokenizer = MockTokenizer()
        
        class TestTokenFeature(TokenFeature):
            def sim_func(self, x, y):
                return 1.0 if x == y else 0.0
            
            def __str__(self):
                return "test_token_feature"
        
        feature = TestTokenFeature("attr1", "attr2", tokenizer)
        
        assert feature._tokenizer == tokenizer
        assert feature._a_toks_col == "tokens(attr1)"
        assert feature._b_toks_col == "tokens(attr2)"
        
        # Test that the feature can be called
        rec = pd.Series({"tokens(attr2)": {"a", "b"}})
        recs = pd.DataFrame({"tokens(attr1)": [{"a", "b"}, {"c", "d"}]})
        result = feature(rec, recs)
        assert len(result) == 2

    def test_token_feature_invalid_tokenizer(self):
        """Test TokenFeature with invalid tokenizer type."""
        with pytest.raises(TypeError, match=r"tokenizer must be type <class 'MadLib._internal.tokenizer.tokenizer.Tokenizer'>"):
            class TestTokenFeature(TokenFeature):
                def sim_func(self, x, y):
                    return 0.0
                
                def __str__(self):
                    return "test_token_feature"
            
            TestTokenFeature("attr1", "attr2", "invalid_tokenizer")

    def test_token_feature_get_input_column(self):
        """Test _get_input_column method."""
        tokenizer = MockTokenizer()
        
        class TestTokenFeature(TokenFeature):
            def sim_func(self, x, y):
                return 0.0
            
            def __str__(self):
                return "test_token_feature"
        
        feature = TestTokenFeature("attr1", "attr2", tokenizer)
        assert feature._get_input_column("test_col") == "test_col"

    def test_token_feature_get_token_column(self):
        """Test _get_token_column method."""
        tokenizer = MockTokenizer()
        
        class TestTokenFeature(TokenFeature):
            def sim_func(self, x, y):
                return 0.0
            
            def __str__(self):
                return "test_token_feature"
        
        feature = TestTokenFeature("attr1", "attr2", tokenizer)
        assert feature._get_token_column("test_col") == "tokens(test_col)"

    def test_token_feature_preprocess_output_column(self):
        """Test _preprocess_output_column method."""
        tokenizer = MockTokenizer()
        
        class TestTokenFeature(TokenFeature):
            def sim_func(self, x, y):
                return 0.0
            
            def __str__(self):
                return "test_token_feature"
        
        feature = TestTokenFeature("attr1", "attr2", tokenizer)
        assert feature._preprocess_output_column("test_col") == "tokens(test_col)"

    def test_token_feature_preprocess(self):
        """Test _preprocess method."""
        tokenizer = MockTokenizer()
        
        class TestTokenFeature(TokenFeature):
            def sim_func(self, x, y):
                return 0.0
            
            def __str__(self):
                return "test_token_feature"
        
        feature = TestTokenFeature("attr1", "attr2", tokenizer)
        
        df = pd.DataFrame({"test_col": ["hello world", "test data"]})
        result = feature._preprocess(df, "test_col")
        
        assert result.name == "tokens(test_col)"
        assert len(result) == 2
        assert result.iloc[0] == {"hello", "world"}
        assert result.iloc[1] == {"test", "data"}

    def test_token_feature_call_with_null_b_tokens(self):
        """Test __call__ method when b_toks_col is null."""
        tokenizer = MockTokenizer()
        
        class TestTokenFeature(TokenFeature):
            def sim_func(self, x, y):
                return 0.0
            
            def __str__(self):
                return "test_token_feature"
        
        feature = TestTokenFeature("attr1", "attr2", tokenizer)
        
        rec = pd.Series({"tokens(attr2)": None})
        recs = pd.DataFrame({"tokens(attr1)": [{"a", "b"}, {"c", "d"}]})
        
        result = feature(rec, recs)
        
        assert len(result) == 2
        assert all(pd.isna(result))

    def test_token_feature_call_normal(self):
        """Test __call__ method with normal data."""
        tokenizer = MockTokenizer()
        
        class TestTokenFeature(TokenFeature):
            def sim_func(self, x, y):
                return 1.0 if x == y else 0.0
            
            def __str__(self):
                return "test_token_feature"
        
        feature = TestTokenFeature("attr1", "attr2", tokenizer)
        
        rec = pd.Series({"tokens(attr2)": {"a", "b"}})
        recs = pd.DataFrame({"tokens(attr1)": [{"a", "b"}, {"c", "d"}]})
        
        result = feature(rec, recs)
        
        assert len(result) == 2
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 0.0
        assert result.dtype == np.float64


@pytest.mark.unit
class TestOverlapFunctions:
    """Test overlap utility functions."""

    def test_overlap_both_null(self):
        """Test _overlap with both inputs null."""
        assert np.isnan(_overlap(None, None))

    def test_overlap_one_null(self):
        """Test _overlap with one input null."""
        assert np.isnan(_overlap(None, {"a", "b"}))
        assert np.isnan(_overlap({"a", "b"}, None))

    def test_overlap_empty_sets(self):
        """Test _overlap with empty sets."""
        assert _overlap(set(), {"a", "b"}) == 0.0
        assert _overlap({"a", "b"}, set()) == 0.0
        assert _overlap(set(), set()) == 0.0

    def test_overlap_no_overlap(self):
        """Test _overlap with no overlapping elements."""
        assert _overlap({"a", "b"}, {"c", "d"}) == 0.0

    def test_overlap_some_overlap(self):
        """Test _overlap with some overlapping elements."""
        assert _overlap({"a", "b", "c"}, {"b", "c", "d"}) == 2.0

    def test_overlap_complete_overlap(self):
        """Test _overlap with complete overlap."""
        assert _overlap({"a", "b"}, {"a", "b"}) == 2.0

    def test_overlap_different_sizes(self):
        """Test _overlap with sets of different sizes."""
        # Should use the smaller set for iteration
        assert _overlap({"a", "b"}, {"a", "b", "c", "d"}) == 2.0
        assert _overlap({"a", "b", "c", "d"}, {"a", "b"}) == 2.0


@pytest.mark.unit
class TestJaccardFunctions:
    """Test Jaccard similarity functions."""

    def test_jaccard_both_null(self):
        """Test _jaccard with both inputs null."""
        assert np.isnan(_jaccard(None, None))

    def test_jaccard_one_null(self):
        """Test _jaccard with one input null."""
        assert np.isnan(_jaccard(None, {"a", "b"}))
        assert np.isnan(_jaccard({"a", "b"}, None))

    def test_jaccard_no_overlap(self):
        """Test _jaccard with no overlapping elements."""
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_some_overlap(self):
        """Test _jaccard with some overlapping elements."""
        # overlap = 2, union = 4, jaccard = 2/4 = 0.5
        assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 0.5

    def test_jaccard_complete_overlap(self):
        """Test _jaccard with complete overlap."""
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_empty_sets(self):
        """Test _jaccard with empty sets."""
        assert _jaccard(set(), {"a", "b"}) == 0.0
        assert _jaccard({"a", "b"}, set()) == 0.0


@pytest.mark.unit
class TestOverlapCoeffFunctions:
    """Test overlap coefficient functions."""

    def test_overlap_coeff_both_null(self):
        """Test _overlap_coeff with both inputs null."""
        assert np.isnan(_overlap_coeff(None, None))

    def test_overlap_coeff_one_null(self):
        """Test _overlap_coeff with one input null."""
        assert np.isnan(_overlap_coeff(None, {"a", "b"}))
        assert np.isnan(_overlap_coeff({"a", "b"}, None))

    def test_overlap_coeff_no_overlap(self):
        """Test _overlap_coeff with no overlapping elements."""
        assert _overlap_coeff({"a", "b"}, {"c", "d"}) == 0.0

    def test_overlap_coeff_some_overlap(self):
        """Test _overlap_coeff with some overlapping elements."""
        # overlap = 2, min_size = 3, coeff = 2/3
        assert _overlap_coeff({"a", "b", "c"}, {"b", "c", "d"}) == 2/3

    def test_overlap_coeff_complete_overlap(self):
        """Test _overlap_coeff with complete overlap."""
        assert _overlap_coeff({"a", "b"}, {"a", "b"}) == 1.0

    def test_overlap_coeff_different_sizes(self):
        """Test _overlap_coeff with sets of different sizes."""
        # overlap = 2, min_size = 2, coeff = 2/2 = 1.0
        assert _overlap_coeff({"a", "b"}, {"a", "b", "c", "d"}) == 1.0


@pytest.mark.unit
class TestJaccardFeature:
    """Test JaccardFeature class."""

    def test_jaccard_feature_creation(self):
        """Test creating JaccardFeature."""
        tokenizer = MockTokenizer()
        feature = JaccardFeature("attr1", "attr2", tokenizer)
        
        assert feature._tokenizer == tokenizer
        assert feature._a_toks_col == "tokens(attr1)"
        assert feature._b_toks_col == "tokens(attr2)"

    def test_jaccard_feature_sim_func(self):
        """Test JaccardFeature sim_func method."""
        tokenizer = MockTokenizer()
        feature = JaccardFeature("attr1", "attr2", tokenizer)
        
        # Test with overlapping sets
        result = feature.sim_func({"a", "b", "c"}, {"b", "c", "d"})
        assert result == 0.5
        
        # Test with no overlap
        result = feature.sim_func({"a", "b"}, {"c", "d"})
        assert result == 0.0
        
        # Test with complete overlap
        result = feature.sim_func({"a", "b"}, {"a", "b"})
        assert result == 1.0

    def test_jaccard_feature_str(self):
        """Test JaccardFeature string representation."""
        tokenizer = MockTokenizer()
        feature = JaccardFeature("attr1", "attr2", tokenizer)
        
        expected = "jaccard(tokens(attr1), tokens(attr2))"
        assert str(feature) == expected


@pytest.mark.unit
class TestOverlapCoeffFeature:
    """Test OverlapCoeffFeature class."""

    def test_overlap_coeff_feature_creation(self):
        """Test creating OverlapCoeffFeature."""
        tokenizer = MockTokenizer()
        feature = OverlapCoeffFeature("attr1", "attr2", tokenizer)
        
        assert feature._tokenizer == tokenizer
        assert feature._a_toks_col == "tokens(attr1)"
        assert feature._b_toks_col == "tokens(attr2)"

    def test_overlap_coeff_feature_sim_func(self):
        """Test OverlapCoeffFeature sim_func method."""
        tokenizer = MockTokenizer()
        feature = OverlapCoeffFeature("attr1", "attr2", tokenizer)
        
        # Test with overlapping sets
        result = feature.sim_func({"a", "b", "c"}, {"b", "c", "d"})
        assert result == 2/3
        
        # Test with no overlap
        result = feature.sim_func({"a", "b"}, {"c", "d"})
        assert result == 0.0
        
        # Test with complete overlap
        result = feature.sim_func({"a", "b"}, {"a", "b"})
        assert result == 1.0

    def test_overlap_coeff_feature_str(self):
        """Test OverlapCoeffFeature string representation."""
        tokenizer = MockTokenizer()
        feature = OverlapCoeffFeature("attr1", "attr2", tokenizer)
        
        expected = "overlap_coeff(tokens(attr1), tokens(attr2))"
        assert str(feature) == expected


@pytest.mark.unit
class TestCosineFeature:
    """Test CosineFeature class."""

    def test_cosine_feature_creation(self):
        """Test creating CosineFeature."""
        tokenizer = MockTokenizer()
        feature = CosineFeature("attr1", "attr2", tokenizer)
        
        assert feature._tokenizer == tokenizer
        assert feature._a_toks_col == "tokens(attr1)"
        assert feature._b_toks_col == "tokens(attr2)"

    def test_cosine_feature_sim_func(self):
        """Test CosineFeature sim_func method."""
        tokenizer = MockTokenizer()
        feature = CosineFeature("attr1", "attr2", tokenizer)
        
        # Test with overlapping sets
        result = feature.sim_func({"a", "b", "c"}, {"b", "c", "d"})
        expected = 2 / (3 ** 0.5 * 3 ** 0.5)  # overlap / sqrt(len(x) * len(y))
        assert abs(result - expected) < 1e-10
        
        # Test with no overlap
        result = feature.sim_func({"a", "b"}, {"c", "d"})
        assert result == 0.0
        
        # Test with complete overlap
        result = feature.sim_func({"a", "b"}, {"a", "b"})
        expected = 2 / (2 ** 0.5 * 2 ** 0.5)
        assert abs(result - expected) < 1e-10

    def test_cosine_feature_str(self):
        """Test CosineFeature string representation."""
        tokenizer = MockTokenizer()
        feature = CosineFeature("attr1", "attr2", tokenizer)
        
        expected = "cosine(tokens(attr1), tokens(attr2))"
        assert str(feature) == expected


@pytest.mark.unit
class TestMongeElkanFeature:
    """Test MongeElkanFeature class."""

    def test_monge_elkan_feature_creation(self):
        """Test creating MongeElkanFeature."""
        tokenizer = MockTokenizer()
        feature = MongeElkanFeature("attr1", "attr2", tokenizer)
        
        assert feature._tokenizer == tokenizer
        assert feature._a_toks_col == "tokens(attr1)"
        assert feature._b_toks_col == "tokens(attr2)"

    def test_monge_elkan_feature_sim_func(self):
        """Test MongeElkanFeature sim_func method."""
        tokenizer = MockTokenizer()
        feature = MongeElkanFeature("attr1", "attr2", tokenizer)
        
        # Mock the _me method directly
        with patch.object(feature, '_me', return_value=0.8) as mock_me:
            # Test with valid inputs (lists)
            result = feature.sim_func(["hello", "world"], ["world", "hello"])
            assert result == 0.8
            mock_me.assert_called_once_with(["hello", "world"], ["world", "hello"])
            
            # Test with null input
            result = feature.sim_func(["hello", "world"], None)
            assert np.isnan(result)

    def test_monge_elkan_feature_str(self):
        """Test MongeElkanFeature string representation."""
        tokenizer = MockTokenizer()
        feature = MongeElkanFeature("attr1", "attr2", tokenizer)
        
        expected = "monge_elkan_jw(attr1, attr2)"
        assert str(feature) == expected


@pytest.mark.unit
class TestTokenFeatureIntegration:
    """Integration tests for token features."""

    def test_token_feature_full_workflow(self):
        """Test complete workflow with a token feature."""
        tokenizer = MockTokenizer()
        feature = JaccardFeature("name", "title", tokenizer)
        
        # Create test data
        df = pd.DataFrame({
            "name": ["hello world", "test data", "python code"],
            "title": ["world hello", "data test", "java script"]
        })
        
        # Preprocess the data
        df["tokens(name)"] = feature._preprocess(df, "name")
        df["tokens(title)"] = feature._preprocess(df, "title")
        
        # Test calling the feature
        rec = df.iloc[0]
        recs = df.iloc[1:]
        
        result = feature(rec, recs)
        
        assert len(result) == 2
        assert all(0 <= score <= 1 for score in result if not pd.isna(score))

    def test_multiple_token_features(self):
        """Test multiple token features together."""
        tokenizer = MockTokenizer()
        
        features = [
            JaccardFeature("name", "title", tokenizer),
            OverlapCoeffFeature("name", "title", tokenizer),
            CosineFeature("name", "title", tokenizer)
        ]
        
        # Test data
        set1 = {"a", "b", "c"}
        set2 = {"b", "c", "d"}
        
        results = [feature.sim_func(set1, set2) for feature in features]
        
        assert len(results) == 3
        assert all(isinstance(r, (int, float)) for r in results)
        assert all(0 <= r <= 1 for r in results) 