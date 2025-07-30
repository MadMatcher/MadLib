# MadLib

A comprehensive Python library for entity matching and record linkage, providing flexible and scalable solutions for matching and linking records across datasets using machine learning and active learning techniques.

## Features

- **Machine Learning Models**: Support for both scikit-learn and PySpark ML models
- **Active Learning**: Efficient labeling with batch and continuous modes
- **Similarity Functions**: TF-IDF, Jaccard, Cosine, SIF embeddings, and more
- **Flexible Labeling**: CLI-based and programmatic labeling interfaces
- **Scalable Processing**: Can leverage PySpark for handling large datasets

## Installation

Install from PyPI: [todo]

```bash
pip install MadLib
```

For development dependencies:

```bash
pip install MadLib
```

### Requirements

**Core Dependencies:**

- Python 3.8+
- numpy>=1.20.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- pyspark>=3.2.0
- pyarrow>=11.0.0
- joblib>=1.1.0
- threadpoolctl>=3.0.0
- scipy>=1.7.0
- numba>=0.56.0
- tqdm>=4.62.0
- tabulate>=0.8.9
- xxhash>=3.0.0
- py-stringmatching>=0.4.0
- mmh3>=3.0.0

## Quick Start

Here's a simple example of using MadLib:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from MadLib import create_features, featurize, create_seeds, train_matcher

# Sample datasets
df_a = pd.DataFrame({
    '_id': [1, 2, 3],
    'name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown'],
    'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com']
})

df_b = pd.DataFrame({
    '_id': [101, 102, 103],
    'name': ['Alicia Smith', 'Robert Johnson', 'Charles Brown'],
    'email': ['alicia@email.com', 'robert@email.com', 'charles@email.com']
})

# Generate candidate pairs
candidates = pd.DataFrame({
    'id1_list': [[1], [2], [3]],
    'id2': [101, 102, 103]
})

# Create features for comparison
features = create_features(df_a, df_b, ['name', 'email'], ['name', 'email'])

# Generate feature vectors
fvs = featurize(features, df_a, df_b, candidates)

# Create labeled seeds for training
gold_labels = pd.DataFrame({'id1': [1, 3], 'id2': [101, 103]})
gold_labeler = {'name': 'gold', 'gold': gold_labels}
seeds = create_seeds(fvs, nseeds=2, labeler=gold_labeler)

# Train a matcher model
model_spec = {
    'model_type': 'sklearn',
    'model': LogisticRegression,
    'nan_fill': 0.0,
    'model_args': {'random_state': 42}
}
trained_model = train_matcher(model_spec, seeds)

# Apply the model to make predictions
predictions = apply_matcher(trained_model, fvs, 'features', 'prediction')
```

## Spark vs Pandas

MadLib supports both Pandas DataFrames and Spark DataFrames, allowing you to choose the best approach for your data size and processing requirements.

### When to Use Spark vs Pandas

**Use Pandas when:**

- Working with smaller datasets
- Prototyping and development
- Simple single-machine processing
- You prefer familiar Pandas syntax

**Use Spark when:**

- Working with large datasets
- Need distributed processing across multiple CPU cores or multiple machines
- Processing data that doesn't fit in memory
- Require scalable, production-ready performance

### Setting Up a SparkSession

MadLib leverages Spark to enhance speed for large-scale record matching. Spark works by processing data in parallel, which significantly increases processing speed. On a local machine, Spark treats each CPU core as a worker node to distribute work efficiently.

Before using Spark DataFrames with MadLib, you need to set up a SparkSession:

```python
from pyspark.sql import SparkSession

# Create a SparkSession for local processing
spark = SparkSession.builder \
    .master('local[*]') \
    .appName('MadLib') \
    .getOrCreate()
```

**Configuration Options:**

- `master('local[*]')`: Uses all available CPU cores on your local machine
- `master('local[4]')`: Uses exactly 4 CPU cores
- `master(url)`: Uses the machine at URL as the driver
- `appName('MadLib')`: Names your Spark application for identification

### Converting Between Pandas and Spark

```python
# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(pandas_df)

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = spark_df.toPandas()
```

### Example with Spark DataFrames

```python
from pyspark.sql import SparkSession
from MadLib import create_features, featurize

# Set up Spark
spark = SparkSession.builder \
    .master('local[*]') \
    .appName('MadLib') \
    .getOrCreate()

# Convert your data to Spark DataFrames
spark_df_a = spark.createDataFrame(df_a)
spark_df_b = spark.createDataFrame(df_b)
spark_candidates = spark.createDataFrame(candidates)

# Use the same MadLib functions
features = create_features(spark_df_a, spark_df_b, ['name', 'email'], ['name', 'email'])
spark_fvs = featurize(features, spark_df_a, spark_df_b, spark_candidates)

# The result is a Spark DataFrame with distributed processing
print(f"Feature vectors count: {spark_fvs.count()}")
```

### Running on a Spark Cluster

For production deployments on a Spark cluster:

```python
from pyspark.sql import SparkSession

# Connect to a Spark cluster
spark = SparkSession.builder \
    .master('spark://your-cluster-master:7077') \
    .appName('MadLib Production') \
    .config('spark.executor.memory', '8g') \
    .config('spark.driver.memory', '4g') \
    .config('spark.sql.adaptive.enabled', 'true') \
    .getOrCreate()

# Your MadLib code remains the same
features = create_features(spark_df_a, spark_df_b, ['name', 'email'], ['name', 'email'])
spark_fvs = featurize(features, spark_df_a, spark_df_b, spark_candidates)
```

For additional SparkSession configuration options (storage limits, memory settings, etc.), refer to the [official PySpark documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.html).

## API Overview

### Core Functions

- **`create_features(A, B, a_cols, b_cols, sim_functions=None, tokenizers=None, null_threshold=0.5)`**: Generate feature objects for comparing records
- **`featurize(features, A, B, candidates, output_col='features', fill_na=0.0)`**: Apply features to candidate pairs
- **`down_sample(fvs, percent, search_id_column, score_column='score', bucket_size=1000)`**: Reduce dataset size by sampling
- **`create_seeds(fvs, nseeds, labeler, score_column='score')`**: Generate labeled training examples
- **`train_matcher(model_spec, labeled_data, feature_col='features', label_col='label')`**: Train a matching model
- **`apply_matcher(model, df, feature_col, output_col)`**: Apply trained model for predictions
- **`label_data(model_spec, mode, labeler_spec, fvs, seeds=None)`**: Generate labeled data using active learning

### Feature Types

- **Exact Match**: Binary exact string matching
- **TF-IDF**: Term frequency-inverse document frequency similarity
- **Jaccard**: Jaccard coefficient for set similarity
- **Cosine**: Cosine similarity between vectors
- **SIF**: Smooth Inverse Frequency embeddings
- **Overlap Coefficient**: Normalized overlap between sets
- **Relative Difference**: Numerical difference normalization

### Model Support

- **Scikit-learn Models**: LogisticRegression, RandomForest, XGBoost, etc.
- **PySpark ML Models**: MLlib classifiers for large-scale processing
- **Custom Models**: Implement the MLModel interface for custom algorithms

### Active Learning

- **Batch Mode**: Process multiple examples at once
- **Continuous Mode**: Interactive labeling with real-time model updates
- **Entropy-based Selection**: Focus on most uncertain examples

## Examples

Check out the comprehensive Jupyter notebook in `examples/madmatcher_examples.ipynb` for detailed usage examples including:

### Table of Contents

1. **Setup** - Installation and imports
2. **Public API Overview** - Complete function reference
3. **Tokenizers and Similarity Functions** - Text preprocessing and comparison methods
4. **Feature Creation** - Building feature objects for record comparison
5. **Featurization** - Converting records to feature vectors
6. **Down Sampling** - Reducing dataset size for efficiency
7. **Seed Creation** - Generating initial labeled examples
8. **Training a Matcher** - Building machine learning models
9. **Applying a Matcher** - Making predictions on new data
10. **Active Learning Labeling** - Efficient data labeling strategies
11. **Custom Abstract Classes** - Extending functionality with custom implementations

For in-depth technical documentation, see [`docs/MadLib Documentation.md`](docs/MadLib%20Documentation.md).

## Documentation

- **API Documentation**: Auto-generated from docstrings
- **Examples**: Comprehensive Jupyter notebook with real examples in [`examples/madmatcher_examples.ipynb`](examples/madmatcher_examples.ipynb)
- **Technical Documentation**: See [`docs/MadLib Documentation.md`](docs/MadLib%20Documentation.md)

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Support

For issues, questions, or feature requests, please:

1. Check the documentation and examples
2. Email us at entitymatchinginfo@gmail.com
