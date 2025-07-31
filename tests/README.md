# MadLib Test Suite

This comprehensive test suite provides thorough testing of the MadLib package with **80% code coverage target** across unit, integration, and performance tests.

## 🏗️ Test Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── pytest.ini              # Pytest settings and markers
├── test_runner.py           # Convenient test execution script
├── unit/                    # Unit tests for individual components
│   ├── test_tools.py        # Public API functions
│   ├── test_tokenizers.py   # Tokenization and similarity functions
│   ├── test_features.py     # Feature creation and featurization
│   ├── test_ml_models.py    # ML model wrappers and custom models
│   ├── test_labelers.py     # Labeling functionality
│   └── test_active_learning.py # Active learning components
├── integration/             # End-to-end workflow tests
│   └── test_end_to_end_workflow.py # Complete pipeline testing
└── performance/             # Performance and scalability tests
    └── test_performance.py  # Memory usage, timing, scaling tests
```

## 🚀 Quick Start

### Prerequisites

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-xdist psutil
```

### Running Tests

Use the test runner script for convenience:

```bash
# Run all tests with coverage
python tests/test_runner.py coverage --html-report

# Run only unit tests
python tests/test_runner.py unit

# Run quick tests (excluding slow ones)
python tests/test_runner.py quick

# Run with verbose output
python tests/test_runner.py all --verbose
```

### Direct pytest commands

```bash
# Run all tests with coverage
pytest --cov=MadLib --cov-report=html --cov-fail-under=80

# Run specific test types
pytest tests/unit/ -m unit
pytest tests/integration/ -m integration
pytest tests/performance/ -m performance

# Run tests in parallel
pytest -n auto tests/unit/

# Run with verbose output
pytest -v tests/
```

## 📊 Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation:

- **`test_tools.py`**: Public API functions (`down_sample`, `create_seeds`, `train_matcher`, `apply_matcher`, `label_data`)
- **`test_tokenizers.py`**: Tokenization functions, similarity calculations, vectorizers
- **`test_features.py`**: Feature creation, custom Feature implementations, featurization workflows
- **`test_ml_models.py`**: SklearnMLModel wrapper, custom MLModel implementations
- **`test_labelers.py`**: Labeler abstract class and implementations
- **`test_active_learning.py`**: Active learning algorithms

**Markers**: `@pytest.mark.unit`

### Integration Tests (`tests/integration/`)

Test complete end-to-end workflows:

- Complete matching pipelines
- Multi-component interactions
- Workflow variations and configurations
- Error handling and robustness
- Data quality scenarios

**Markers**: `@pytest.mark.integration`

### Performance Tests (`tests/performance/`)

Test scalability and efficiency:

- Performance scaling with dataset size
- Memory usage optimization
- Concurrent execution scenarios
- Performance regression detection
- Baseline benchmarks

**Markers**: `@pytest.mark.performance`, `@pytest.mark.slow`

## 🧪 Test Fixtures

### Core Fixtures (in `conftest.py`)

- **`spark_session`**: Spark session for testing (session scope)
- **`sample_dataframe_a/b`**: Representative DataFrames for testing
- **`sample_candidates`**: Basic candidate pairs
- **`sample_blocked_candidates`**: Blocked candidate format
- **`gold_labels`**: Ground truth labels for evaluation
- **`sample_feature_vectors`**: Pre-computed feature vectors
- **`mock_labeled_data`**: Training data for ML models
- **`test_data_generator`**: Utility for creating test datasets
- **`performance_timer`**: Timer for performance measurements

### Usage Example

```python
def test_my_function(sample_dataframe_a, sample_dataframe_b, sample_candidates):
    """Test using provided fixtures."""
    result = my_function(sample_dataframe_a, sample_dataframe_b, sample_candidates)
    assert len(result) == len(sample_candidates)
```

## 📈 Coverage Requirements

The test suite targets **80% code coverage** with the following expectations:

- **Unit tests**: 90%+ coverage of individual modules
- **Integration tests**: Cover all public API workflows
- **Performance tests**: Ensure scalability requirements
- **Edge cases**: Handle error conditions and boundary cases

### Coverage Reports

Generate coverage reports:

```bash
# Terminal report
pytest --cov=MadLib --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=MadLib --cov-report=html
open htmlcov/index.html

# XML report (for CI/CD)
pytest --cov=MadLib --cov-report=xml
```

## 🏷️ Test Markers

Use pytest markers to organize and run specific test subsets:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Exclude slow tests
pytest -m "not slow"

# Run unit and integration, but not performance
pytest -m "unit or integration"
```

### Available Markers

- `unit`: Unit tests for individual components
- `integration`: Integration tests for complete workflows
- `performance`: Performance and scalability tests
- `slow`: Tests that take longer to execute

## 🔧 Customization

### Adding New Tests

1. **Unit tests**: Add to appropriate `test_*.py` file in `tests/unit/`
2. **Integration tests**: Add to `test_end_to_end_workflow.py` or create new file
3. **Performance tests**: Add to `test_performance.py`

### Test Templates

#### Unit Test Template

```python
import pytest
from MadLib import your_module

@pytest.mark.unit
class TestYourModule:
    """Test YourModule functionality."""

    def test_basic_functionality(self, sample_fixture):
        """Test basic functionality."""
        result = your_module.function(sample_fixture)
        assert result is not None

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            your_module.function(invalid_input)
```

#### Integration Test Template

```python
import pytest
from MadLib.tools import *

@pytest.mark.integration
class TestYourWorkflow:
    """Test complete workflow."""

    def test_complete_pipeline(self, sample_dataframe_a, sample_dataframe_b):
        """Test end-to-end pipeline."""
        # Step 1: Feature creation
        # Step 2: Training
        # Step 3: Prediction
        # Step 4: Validation
        pass
```

### Custom Fixtures

Add custom fixtures to `conftest.py`:

```python
@pytest.fixture
def custom_test_data():
    """Create custom test data."""
    return create_custom_data()
```

## 🚀 Continuous Integration

### CI Configuration

For GitHub Actions, CircleCI, etc.:

```yaml
- name: Run tests
  run: |
    python tests/test_runner.py coverage --html-report

- name: Upload coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```

### Pre-commit Hooks

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: pytest-quick
      name: pytest-quick
      entry: python tests/test_runner.py quick
      language: system
      pass_filenames: false
```

## 🐛 Debugging Tests

### Common Issues

1. **Import errors**: Ensure package is installed in development mode
2. **Fixture not found**: Check fixture names and imports
3. **Spark issues**: Verify Spark session configuration
4. **Performance test failures**: May be environment-dependent

### Debug Commands

```bash
# Run with verbose output and stop on first failure
pytest -vvv -x tests/unit/test_tools.py::TestDownSample::test_basic

# Run specific test with prints
pytest -s tests/unit/test_tools.py::TestDownSample::test_basic

# Debug with pdb
pytest --pdb tests/unit/test_tools.py::TestDownSample::test_basic
```

## 📝 Best Practices

### Writing Tests

1. **Use descriptive test names**: `test_featurize_with_missing_data`
2. **Test edge cases**: Empty inputs, invalid parameters, boundary conditions
3. **Use appropriate fixtures**: Leverage shared test data
4. **Mark tests properly**: Use `@pytest.mark.unit`, etc.
5. **Keep tests focused**: One concept per test
6. **Mock external dependencies**: Use `unittest.mock` for isolation

### Test Organization

1. **Group related tests**: Use test classes
2. **Follow naming conventions**: `test_*.py`, `Test*`, `test_*`
3. **Document test purpose**: Clear docstrings
4. **Maintain test data**: Keep fixtures up to date

### Performance Testing

1. **Set realistic thresholds**: Based on expected usage
2. **Account for environment**: CI may be slower than local
3. **Use relative benchmarks**: Compare to baseline, not absolute time
4. **Monitor memory usage**: Prevent memory leaks

## 📞 Support

For questions about the test suite:

1. Check existing test examples
2. Review fixture documentation in `conftest.py`
3. Run tests with `-v` flag for detailed output
4. Use `pytest --collect-only` to see available tests

## 🎯 Coverage Goals

Current coverage targets by module:

- `tools.py`: 95%
- `_internal/`: 85%
- Abstract classes: 90%
- Integration workflows: 100%

Run `python tests/test_runner.py coverage --html-report` to see detailed coverage by file and line.
