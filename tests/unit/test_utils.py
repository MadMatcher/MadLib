"""Unit tests for utils module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark import StorageLevel

from madmatcher_tools._internal.utils import (
    type_check, type_check_iterable, is_null, persisted, is_persisted,
    get_logger, PerfectHashFunction, SparseVec
)


@pytest.mark.unit
class TestTypeCheck:
    """Test type checking utilities."""

    def test_type_check_valid(self):
        """Test type_check with valid type."""
        # Should not raise an exception
        type_check(42, "test_var", int)
        type_check("hello", "test_var", str)
        type_check([1, 2, 3], "test_var", list)

    def test_type_check_invalid(self):
        """Test type_check with invalid type."""
        with pytest.raises(TypeError, match="test_var must be type <class 'int'>"):
            type_check("not_an_int", "test_var", int)
        
        with pytest.raises(TypeError, match="test_var must be type <class 'str'>"):
            type_check(42, "test_var", str)

    def test_type_check_iterable_valid(self):
        """Test type_check_iterable with valid iterable."""
        # Should not raise an exception
        type_check_iterable([1, 2, 3], "test_var", list, int)
        type_check_iterable((1, 2, 3), "test_var", tuple, int)
        type_check_iterable(["a", "b", "c"], "test_var", list, str)

    def test_type_check_iterable_invalid_container(self):
        """Test type_check_iterable with invalid container type."""
        with pytest.raises(TypeError, match="test_var must be type <class 'list'>"):
            type_check_iterable((1, 2, 3), "test_var", list, int)

    def test_type_check_iterable_invalid_elements(self):
        """Test type_check_iterable with invalid element types."""
        with pytest.raises(TypeError, match="all elements of test_var must be type<class 'int'>"):
            type_check_iterable([1, "not_int", 3], "test_var", list, int)


@pytest.mark.unit
class TestIsNull:
    """Test is_null function."""

    def test_is_null_with_none(self):
        """Test is_null with None."""
        assert is_null(None) is True

    def test_is_null_with_nan(self):
        """Test is_null with NaN."""
        assert is_null(np.nan) is True
        assert is_null(float('nan')) is True

    def test_is_null_with_valid_values(self):
        """Test is_null with valid values."""
        assert is_null(42) is False
        assert is_null("hello") is False
        assert is_null(0) is False
        assert is_null("") is False
        assert is_null([]) is False

    def test_is_null_with_pandas_series(self):
        """Test is_null with pandas Series."""
        series = pd.Series([1, np.nan, 3, None])
        # is_null should return False for pandas Series (not null itself)
        assert is_null(series) is False


@pytest.mark.unit
class TestPersisted:
    """Test persisted context manager."""

    def test_persisted_with_dataframe(self, spark_session):
        """Test persisted with Spark DataFrame."""
        data = [(1, "hello"), (2, "world")]
        schema = StructType([
            StructField("_id", IntegerType(), False),
            StructField("text", StringType(), False)
        ])
        df = spark_session.createDataFrame(data, schema)
        
        with persisted(df) as persisted_df:
            assert persisted_df is not None
            assert persisted_df.count() == 2

    def test_persisted_with_none(self):
        """Test persisted with None."""
        with persisted(None) as result:
            assert result is None

    def test_persisted_with_custom_storage_level(self, spark_session):
        """Test persisted with custom storage level."""
        data = [(1, "hello"), (2, "world")]
        schema = StructType([
            StructField("_id", IntegerType(), False),
            StructField("text", StringType(), False)
        ])
        df = spark_session.createDataFrame(data, schema)
        
        with persisted(df, StorageLevel.DISK_ONLY) as persisted_df:
            assert persisted_df is not None
            assert persisted_df.count() == 2


@pytest.mark.unit
class TestIsPersisted:
    """Test is_persisted function."""

    def test_is_persisted_with_persisted_dataframe(self, spark_session):
        """Test is_persisted with persisted DataFrame."""
        data = [(1, "hello"), (2, "world")]
        schema = StructType([
            StructField("_id", IntegerType(), False),
            StructField("text", StringType(), False)
        ])
        df = spark_session.createDataFrame(data, schema)
        
        # Persist the DataFrame
        persisted_df = df.persist(StorageLevel.MEMORY_AND_DISK)
        
        assert is_persisted(persisted_df) is True
        
        # Clean up
        persisted_df.unpersist()

    def test_is_persisted_with_unpersisted_dataframe(self, spark_session):
        """Test is_persisted with unpersisted DataFrame."""
        data = [(1, "hello"), (2, "world")]
        schema = StructType([
            StructField("_id", IntegerType(), False),
            StructField("text", StringType(), False)
        ])
        df = spark_session.createDataFrame(data, schema)
        
        assert is_persisted(df) is False


@pytest.mark.unit
class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_basic(self):
        """Test get_logger with basic parameters."""
        logger = get_logger("test_module")
        
        assert logger.name == "test_module"
        assert logger.level == 10  # DEBUG level

    def test_get_logger_with_custom_level(self):
        """Test get_logger with custom level."""
        logger = get_logger("test_module", level=20)  # INFO level
        
        assert logger.name == "test_module"
        assert logger.level == 20

    def test_get_logger_same_name_returns_same_logger(self):
        """Test that get_logger returns the same logger for the same name."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")
        
        assert logger1 is logger2


@pytest.mark.unit
class TestPerfectHashFunction:
    """Test PerfectHashFunction class."""

    def test_perfect_hash_function_creation(self):
        """Test creating PerfectHashFunction."""
        phf = PerfectHashFunction()
        assert phf._seed is not None

    def test_perfect_hash_function_with_seed(self):
        """Test creating PerfectHashFunction with specific seed."""
        seed = 12345
        phf = PerfectHashFunction(seed)
        assert phf._seed == seed

    def test_perfect_hash_function_hash(self):
        """Test hash method."""
        phf = PerfectHashFunction()
        
        # Test with different types of input
        assert isinstance(phf.hash("hello"), int)
        assert isinstance(phf.hash("world"), int)
        assert isinstance(phf.hash("test"), int)
        
        # Same input should produce same hash
        assert phf.hash("hello") == phf.hash("hello")

    def test_perfect_hash_function_create_for_keys(self):
        """Test create_for_keys class method."""
        keys = ["key1", "key2", "key3", "key4"]
        
        phf, hash_vals = PerfectHashFunction.create_for_keys(keys)
        
        assert isinstance(phf, PerfectHashFunction)
        assert isinstance(hash_vals, np.ndarray)
        assert len(hash_vals) == len(keys)
        assert hash_vals.dtype == np.int64

    def test_perfect_hash_function_create_for_keys_duplicates(self):
        """Test create_for_keys with duplicate keys."""
        keys = ["key1", "key2", "key1"]  # Duplicate key
        
        with pytest.raises(ValueError, match="keys must be unique"):
            PerfectHashFunction.create_for_keys(keys)

    def test_perfect_hash_function_create_for_keys_empty(self):
        """Test create_for_keys with empty keys."""
        keys = []
        
        phf, hash_vals = PerfectHashFunction.create_for_keys(keys)
        
        assert isinstance(phf, PerfectHashFunction)
        assert isinstance(hash_vals, np.ndarray)
        assert len(hash_vals) == 0


@pytest.mark.unit
class TestSparseVec:
    """Test SparseVec class."""

    def test_sparse_vec_creation(self):
        """Test creating SparseVec."""
        indices = np.array([0, 2, 4])
        values = np.array([1.0, 2.0, 3.0])
        size = 5
        
        sparse_vec = SparseVec(size, indices, values)
        
        assert sparse_vec._size == size
        np.testing.assert_array_equal(sparse_vec._indexes, indices.astype(np.int32))
        np.testing.assert_array_equal(sparse_vec._values, values.astype(np.float32))

    def test_sparse_vec_to_dense(self):
        """Test sparse vector properties and dot product instead of to_dense."""
        indices = np.array([0, 2, 4])
        values = np.array([1.0, 2.0, 3.0])
        size = 5
        
        sparse_vec = SparseVec(size, indices, values)
        
        # Test properties instead of to_dense
        np.testing.assert_array_equal(sparse_vec.indexes, indices.astype(np.int32))
        np.testing.assert_array_equal(sparse_vec.values, values.astype(np.float32))
        
        # Test dot product with itself
        result = sparse_vec.dot(sparse_vec)
        expected = 1.0**2 + 2.0**2 + 3.0**2  # dot product with itself
        assert result == expected

    def test_sparse_vec_to_dense_larger_size(self):
        """Test sparse vector with larger size using dot product."""
        indices = np.array([0, 2])
        values = np.array([1.0, 2.0])
        size = 10
        
        sparse_vec = SparseVec(size, indices, values)
        
        # Test properties
        np.testing.assert_array_equal(sparse_vec.indexes, indices.astype(np.int32))
        np.testing.assert_array_equal(sparse_vec.values, values.astype(np.float32))
        
        # Test dot product with another sparse vector
        other_indices = np.array([1, 2, 3])
        other_values = np.array([1.0, 1.0, 1.0])
        other_sparse_vec = SparseVec(size, other_indices, other_values)
        
        result = sparse_vec.dot(other_sparse_vec)
        # Only index 2 overlaps: 2.0 * 1.0 = 2.0
        assert result == 2.0

    def test_sparse_vec_to_dense_empty(self):
        """Test empty sparse vector using dot product."""
        indices = np.array([])
        values = np.array([])
        size = 5
        
        sparse_vec = SparseVec(size, indices, values)
        
        # Test properties
        np.testing.assert_array_equal(sparse_vec.indexes, indices.astype(np.int32))
        np.testing.assert_array_equal(sparse_vec.values, values.astype(np.float32))
        
        # Test dot product with non-empty vector
        other_indices = np.array([1, 3])
        other_values = np.array([1.0, 1.0])
        other_sparse_vec = SparseVec(size, other_indices, other_values)
        
        result = sparse_vec.dot(other_sparse_vec)
        # Empty vector dot product should be 0.0
        assert result == 0.0

    def test_sparse_vec_dot_product(self):
        """Test dot product method."""
        indices1 = np.array([0, 2, 4])
        values1 = np.array([1.0, 2.0, 3.0])
        size = 5
        
        indices2 = np.array([1, 2, 3])
        values2 = np.array([1.0, 1.0, 1.0])
        
        sparse_vec1 = SparseVec(size, indices1, values1)
        sparse_vec2 = SparseVec(size, indices2, values2)
        
        # Only index 2 overlaps, so dot product should be 2.0 * 1.0 = 2.0
        result = sparse_vec1.dot(sparse_vec2)
        assert result == 2.0

    def test_sparse_vec_dot_product_no_overlap(self):
        """Test dot product with no overlapping indices."""
        indices1 = np.array([0, 2])
        values1 = np.array([1.0, 2.0])
        size = 5
        
        indices2 = np.array([1, 3])
        values2 = np.array([1.0, 1.0])
        
        sparse_vec1 = SparseVec(size, indices1, values1)
        sparse_vec2 = SparseVec(size, indices2, values2)
        
        # No overlapping indices, so dot product should be 0.0
        result = sparse_vec1.dot(sparse_vec2)
        assert result == 0.0

    def test_sparse_vec_dot_product_empty(self):
        """Test dot product with empty sparse vector."""
        indices1 = np.array([])
        values1 = np.array([])
        size = 5
        
        indices2 = np.array([1, 3])
        values2 = np.array([1.0, 1.0])
        
        sparse_vec1 = SparseVec(size, indices1, values1)
        sparse_vec2 = SparseVec(size, indices2, values2)
        
        # Empty vector dot product should be 0.0
        result = sparse_vec1.dot(sparse_vec2)
        assert result == 0.0

    def test_sparse_vec_properties(self):
        """Test SparseVec properties."""
        indices = np.array([0, 2, 4])
        values = np.array([1.0, 2.0, 3.0])
        size = 5
        
        sparse_vec = SparseVec(size, indices, values)
        
        np.testing.assert_array_equal(sparse_vec.indexes, indices.astype(np.int32))
        np.testing.assert_array_equal(sparse_vec.values, values.astype(np.float32))


@pytest.mark.unit
class TestUtilsIntegration:
    """Test integration scenarios for utils."""

    def test_type_checking_integration(self):
        """Test type checking in a realistic scenario."""
        def process_numbers(numbers):
            type_check_iterable(numbers, "numbers", list, int)
            return sum(numbers)
        
        # Should work
        assert process_numbers([1, 2, 3]) == 6
        
        # Should fail
        with pytest.raises(TypeError):
            process_numbers([1, "not_a_number", 3])

    def test_logging_integration(self):
        """Test logging in a realistic scenario."""
        logger = get_logger("integration_test")
        logger.info("Test message")
        # Should not raise any exceptions

    def test_hash_function_integration(self):
        """Test hash function in a realistic scenario."""
        keys = ["apple", "banana", "cherry"]
        phf, hash_vals = PerfectHashFunction.create_for_keys(keys)
        
        # Should return valid hashes
        for key in keys:
            assert isinstance(phf.hash(key), int)

    def test_sparse_vector_integration(self):
        """Test sparse vector in a realistic scenario."""
        vec1 = SparseVec(5, np.array([1, 3]), np.array([2.0, 4.0]))
        vec2 = SparseVec(5, np.array([1, 2]), np.array([1.0, 3.0]))
        
        # Should compute dot product correctly
        dot_product = vec1.dot(vec2)
        assert dot_product == 2.0  # 2.0 * 1.0 = 2.0

    def test_repartition_df_basic(self):
        """Test repartition_df function."""
        from madmatcher_tools._internal.utils import repartition_df
        # Mock Spark DataFrame
        mock_df = MagicMock()
        mock_df.count.return_value = 1000
        mock_df.repartition.return_value = mock_df
        # No 'by' column
        out = repartition_df(mock_df, 100)
        assert out is mock_df
        # With 'by' column
        out2 = repartition_df(mock_df, 100, by='col')
        assert out2 is mock_df 