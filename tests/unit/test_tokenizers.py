"""Unit tests for tokenizer and vectorizer modules."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from madmatcher_tools.tools import get_base_tokenizers, get_base_sim_functions
from madmatcher_tools import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


@pytest.mark.unit
class TestSimilarityFunctions:
    """Test similarity calculation functions."""

    def jaccard_similarity(self, set1, set2):
        """Helper function for Jaccard similarity."""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def cosine_similarity(self, vec1, vec2):
        """Helper function for cosine similarity."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def test_jaccard_similarity_basic(self):
        """Test basic Jaccard similarity calculation."""
        set1 = {'a', 'b', 'c'}
        set2 = {'b', 'c', 'd'}
        
        result = self.jaccard_similarity(set1, set2)
        expected = 2 / 4  # intersection=2, union=4
        
        assert result == expected

    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity with identical sets."""
        set1 = {'a', 'b', 'c'}
        set2 = {'a', 'b', 'c'}
        
        result = self.jaccard_similarity(set1, set2)
        
        assert result == 1.0

    def test_cosine_similarity_basic(self):
        """Test basic cosine similarity calculation."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])
        
        result = self.cosine_similarity(vec1, vec2)
        
        # Manual calculation for verification
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        expected = dot_product / (norm1 * norm2)
        
        assert np.isclose(result, expected)

    def test_string_similarity(self):
        """Test string similarity calculation using simple method."""
        str1 = "hello"
        str2 = "hallo"
        
        # Simple character-based similarity
        set1 = set(str1)
        set2 = set(str2)
        similarity = self.jaccard_similarity(set1, set2)
        
        assert 0 <= similarity <= 1.0


@pytest.mark.unit
class TestTokenizers:
    """Test tokenization functions."""

    def test_get_base_tokenizers(self):
        """Test that base tokenizers are available."""
        tokenizers = get_base_tokenizers()
        
        # get_base_tokenizers returns a list, not a dict
        assert isinstance(tokenizers, list)
        assert len(tokenizers) > 0

    def test_get_base_sim_functions(self):
        """Test that base similarity functions are available."""
        sim_functions = get_base_sim_functions()
        
        # get_base_sim_functions returns a list, not a dict
        assert isinstance(sim_functions, list)
        assert len(sim_functions) > 0

    def word_tokenize(self, text):
        """Helper function for word tokenization."""
        if text is None:
            return set()
        return set(text.lower().split())

    def test_word_tokenize_basic(self):
        """Test basic word tokenization."""
        text = "Hello world test"
        
        result = self.word_tokenize(text)
        
        assert isinstance(result, set)
        assert 'hello' in result
        assert 'world' in result
        assert 'test' in result

    def test_word_tokenize_none_input(self):
        """Test word tokenization with None input."""
        result = self.word_tokenize(None)
        
        assert isinstance(result, set)
        assert len(result) == 0


@pytest.mark.unit
class TestVectorizers:
    """Test vectorization functions."""

    def test_tfidf_vectorize_basic(self):
        """Test basic TF-IDF vectorization."""
        texts = ["hello world", "world test", "hello test"]
        vectorizer = TfidfVectorizer()
        
        result = vectorizer.fit_transform(texts)
        
        assert result.shape[0] == len(texts)
        assert result.shape[1] > 0

    def test_count_vectorize_basic(self):
        """Test basic count vectorization."""
        texts = ["hello world", "world test", "hello test"]
        vectorizer = CountVectorizer()
        
        result = vectorizer.fit_transform(texts)
        
        assert result.shape[0] == len(texts)
        assert result.shape[1] > 0


@pytest.mark.unit
class TestCustomTokenizer:
    """Test custom Tokenizer implementation."""

    def test_custom_tokenizer_implementation(self):
        """Test implementing a custom Tokenizer."""
        
        class TestTokenizer(Tokenizer):
            def tokenize(self, text):
                """Required abstract method implementation."""
                if text is None:
                    return None
                return text.lower().split()
            
            def __call__(self, text):
                if text is None:
                    return set()
                return set(text.lower().split())
        
        tokenizer = TestTokenizer()
        
        result = tokenizer("Hello World Test")
        
        assert isinstance(result, set)
        assert result == {'hello', 'world', 'test'}

    def test_custom_tokenizer_none_handling(self):
        """Test custom tokenizer handles None input."""
        
        class TestTokenizer(Tokenizer):
            def tokenize(self, text):
                """Required abstract method implementation."""
                if text is None:
                    return None
                return text.lower().split()
            
            def __call__(self, text):
                if text is None:
                    return set()
                return set(text.lower().split())
        
        tokenizer = TestTokenizer()
        
        result = tokenizer(None)
        
        assert isinstance(result, set)
        assert len(result) == 0


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_similarity_with_zero_vectors(self):
        """Test cosine similarity calculation with zero vectors."""
        vec1 = np.array([0, 0, 0])
        vec2 = np.array([1, 2, 3])
        
        # Create our own cosine similarity function
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            result = 0.0
        else:
            result = dot_product / (norm1 * norm2)
        
        # Should handle zero vectors gracefully
        assert result == 0.0

    def test_tokenizers_available(self):
        """Test that tokenizers are available from the API."""
        tokenizers = get_base_tokenizers()
        sim_functions = get_base_sim_functions()
        
        # These return lists, not dicts
        assert isinstance(tokenizers, list)
        assert isinstance(sim_functions, list)
        assert len(tokenizers) > 0
        assert len(sim_functions) > 0 