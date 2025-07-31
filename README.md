# MadLib

A comprehensive Python library for entity matching and record linkage, providing flexible and scalable solutions for matching and linking records across datasets using machine learning and active learning techniques.

## Features

- **Machine Learning Models**: Support for both Scikit-Learn and PySpark ML models
- **Active Learning**: Efficient labeling with batch and continuous modes
- **Similarity Functions**: TF-IDF, Jaccard, Overlap, SIF embeddings, and Cosine Similarity
- **Flexible Labeling**: CLI-based, Web-based, and Gold-based labeling interfaces
- **Scalable Processing**: Can leverage PySpark for handling large datasets

## Installation

Install from PyPI: [todo]

```bash
pip install MadLib
```

Install from GitHub:

```bash
pip install git+https://github.com/dahluwalia/madlib.git
```

### Requirements

**Core Dependencies:**

- Python 3.8+
- flask>=2.0.0
- joblib==1.2.0
- mmh3==3.0.0
- numba==0.60.0
- numpy==1.26.0
- numpydoc==1.5.0
- pandas==1.5.2
- pyarrow==14.0.0
- py_stringmatching==0.4.6
- pyspark==3.5.3
- requests>=2.25.0
- scikit_learn==1.3.1
- scipy==1.15.2
- streamlit>=1.0.0
- tabulate>=0.8.9
- threadpoolctl==3.1.0
- tqdm==4.64.1
- xgboost==1.7.3
- xxhash>=3.0.0

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
- You prefer Pandas syntax

**Use Spark when:**

- Working with large datasets
- Processing data that doesn't fit in memory
- You prefer familiar Spark syntax

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

<!---
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
--->

For additional SparkSession configuration options (storage limits, memory settings, etc.), refer to the [official PySpark documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.html).

## API Overview

### Core Functions

- **`create_features(A, B, a_cols, b_cols, sim_functions=None, tokenizers=None, null_threshold=0.5)`**: Generate feature objects for comparing records
- **`featurize(features, A, B, candidates, output_col='features', fill_na=0.0)`**: Apply features to candidate pairs and obtain feature vectors
- **`down_sample(fvs, percent, search_id_column, score_column='score', bucket_size=1000)`**: Reduce dataset size by intelligently sampling to get matches and non-matches
- **`create_seeds(fvs, nseeds, labeler, score_column='score')`**: Create labeled training examples
- **`train_matcher(model_spec, labeled_data, feature_col='features', label_col='label')`**: Train a matching model
- **`apply_matcher(model, df, feature_col, output_col)`**: Apply trained model for predictions
- **`label_data(model_spec, mode, labeler_spec, fvs, seeds=None)`**: Created labeled data using active learning and only label the most informative pairs

### Model Support

- **Scikit-learn Models**: LogisticRegression, RandomForest, XGBoost, etc.
- **PySpark ML Models**: MLlib classifiers for large-scale processing
- **Custom Models**: Implement the MLModel interface for custom algorithms

### Active Learning

#### **Entropy-based Selection**: Focus on most uncertain examples

- **Batch Mode**: Label in batches before training a new model
- **Continuous Mode**: Continuously label data as new models are being trained at the same time

## Examples

Check out the comprehensive [Python notebook](https://github.com/dahluwalia/MadLib/blob/main/examples/madlib_examples.ipynb) for detailed usage examples including:

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
    
For an overfiew of the API, see the [API Docs todo when live]().
For in-depth API documentation and explanations, see the [MadLib Technical Guide](https://github.com/dahluwalia/MadLib/blob/main/docs/MadLib-Technical-Guide.md).

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Support

For issues, questions, or feature requests, please:

1. Check the documentation and examples
2. Email us at entitymatchinginfo@gmail.com
