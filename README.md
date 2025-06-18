# madmatcher-tools

Tools for entity matching and record linkage, providing a flexible and scalable solution for matching and linking records across datasets.

## Features

- Machine learning models for entity matching (both scikit-learn and PySpark)
- Active learning for efficient labeling
- Feature engineering and vectorization
- CLI-based labeling interface

## Installation

## Quick Start

Here's a simple example of using madmatcher-tools:

## Testing

### Running Tests

To run all tests:

```bash
python -m pytest tests/unit
```

To run tests with verbose output:

```bash
python -m pytest tests/unit -v
```

To run a specific test file:

```bash
python -m pytest tests/unit/test_tools.py
```

To run a specific test class:

```bash
python -m pytest tests/unit/test_tools.py::TestCreateSeeds
```

To run a specific test method:

```bash
python -m pytest tests/unit/test_tools.py::TestCreateSeeds::test_create_seeds_gold_labeler
```

### Test Coverage

To run tests with coverage reporting:

```bash
python -m pytest tests/unit --cov=madmatcher_tools --cov-report=term
```

To generate a detailed coverage report:

```bash
python -m pytest tests/unit --cov=madmatcher_tools --cov-report=term-missing
```

To save coverage report to file:

```bash
python -m pytest tests/unit --cov=madmatcher_tools --cov-report=term > tests/coverage.txt
```

### Test Structure

The test suite is organized as follows:

- **`tests/unit/`** - Unit tests for all modules
  - `test_tools.py` - Tests for the main tools API
  - `test_ml_model.py` - Tests for machine learning models
  - `test_feature_base.py` - Unit tests for feature classes
  - `test_features.py` - Integration tests for feature creation and featurization
  - `test_utils.py` - Tests for utility functions
  - `test_storage.py` - Tests for storage components
  - `test_labelers.py` - Tests for labeling components
  - `test_active_learning.py` - Tests for active learning
  - `test_tokenizers.py` - Tests for tokenization
  - `test_api_utils.py` - Tests for API utilities

### Test Requirements

Tests require:

- Python 3.8+
- pytest
- pytest-cov
- pandas
- numpy
- scikit-learn
- pyspark (for Spark-related tests)

### Coverage Goals

The project maintains **81%+ test coverage** with all tests passing. Coverage reports are generated in the `tests/` directory.

## Documentation

For detailed documentation, visit our [Read the Docs page]().

## License
