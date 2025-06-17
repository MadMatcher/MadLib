"""
Pytest configuration and shared fixtures for MadMatcher tests.
"""

import pytest
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def spark_session():
    """Create a Spark session for testing."""
    spark = SparkSession.builder \
        .appName("MadMatcherTests") \
        .master("local[2]") \
        .config("spark.sql.warehouse.dir", tempfile.mkdtemp()) \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    
    # Set log level to reduce noise during testing
    spark.sparkContext.setLogLevel("WARN")
    
    yield spark
    spark.stop()


@pytest.fixture
def sample_dataframe_a():
    """Sample DataFrame A for testing."""
    return pd.DataFrame({
        '_id': [1, 2, 3, 4, 5],
        'name': ['Alice Smith', 'Bob Jones', 'Carol Davis', 'David Wilson', 'Eve Brown'],
        'age': [25, 30, 28, 35, 22],
        'email': ['alice@email.com', 'bob@email.com', 'carol@email.com', 'david@email.com', None],
        'phone': ['123-456-7890', '987-654-3210', '555-123-4567', None, '111-222-3333'],
        'address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '555 Park Ave']
    })


@pytest.fixture
def sample_dataframe_b():
    """Sample DataFrame B for testing."""
    return pd.DataFrame({
        '_id': [101, 102, 103, 104, 105],
        'name': ['Alicia Smith', 'Robert Jones', 'Caroline Davis', 'Dave Wilson', 'Eva Brown'],
        'age': [26, 29, 28, 36, 23],
        'email': ['alicia@email.com', 'robert@gmail.com', 'caroline@email.com', None, 'eva@email.com'],
        'phone': ['123-456-7891', '987-654-3211', None, '555-999-8888', '111-222-3334'],
        'address': ['124 Main St', '457 Oak Ave', '790 Pine Rd', '322 Elm St', '556 Park Ave']
    })


@pytest.fixture
def sample_candidates():
    """Sample candidate pairs for testing."""
    return pd.DataFrame({
        'id1_list': [[1], [2], [3], [4], [5]],
        'id2': [101, 102, 103, 104, 105]
    })


@pytest.fixture
def sample_blocked_candidates():
    """Sample blocked candidate pairs for testing."""
    return pd.DataFrame({
        'id1_list': [[1, 2], [3], [4, 5]],
        'id2': [101, 103, 104]
    })


@pytest.fixture
def gold_labels():
    """Sample gold standard labels for testing."""
    return pd.DataFrame({
        'id1': [1, 3, 5],
        'id2': [101, 103, 105]
    })


@pytest.fixture
def sample_feature_vectors():
    """Sample feature vectors for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'id1': [1, 2, 3, 4, 5],
        'id2': [101, 102, 103, 104, 105],
        'features': [np.random.random(10).tolist() for _ in range(5)],
        'score': np.random.beta(2, 5, 5),
        '_id': range(5)
    })


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_labeled_data(sample_feature_vectors, gold_labels):
    """Create mock labeled data for testing."""
    labeled = sample_feature_vectors.copy()
    # Add labels based on gold standard
    labeled['label'] = 0.0
    for _, row in gold_labels.iterrows():
        mask = (labeled['id1'] == row['id1']) & (labeled['id2'] == row['id2'])
        labeled.loc[mask, 'label'] = 1.0
    return labeled


# Test data generation utilities
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_large_dataframe(n_rows=1000, missing_rate=0.1):
        """Create a large DataFrame for performance testing."""
        np.random.seed(42)
        names = [f"Person_{i}" for i in range(n_rows)]
        ages = np.random.randint(18, 80, n_rows)
        emails = [f"person{i}@email.com" if np.random.random() > missing_rate 
                 else None for i in range(n_rows)]
        
        return pd.DataFrame({
            '_id': range(1, n_rows + 1),
            'name': names,
            'age': ages,
            'email': emails
        })
    
    @staticmethod
    def create_feature_vector_with_nans(length=10, nan_rate=0.2):
        """Create a feature vector with some NaN values."""
        np.random.seed(42)
        vector = np.random.random(length)
        nan_mask = np.random.random(length) < nan_rate
        vector[nan_mask] = np.nan
        return vector.tolist()


@pytest.fixture
def test_data_generator():
    """Provide access to test data generator."""
    return TestDataGenerator()


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.end_time - self.start_time
        
        @property
        def elapsed(self):
            if self.end_time and self.start_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    ) 