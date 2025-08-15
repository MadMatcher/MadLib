## MadLib: A Library of EM Functions

### Introduction

When performing entity matching (EM), *users often want to experiment with a variety of EM workflows and run these workflows in a variety of runtime environments.* MadLib addresses these needs. It is an open-source library of EM functions that can be combined to create a variety of EM workflows, for a variety of runtime environments. MadLib focuses on the workflows in the matching step (but will also support workflows in the blocking step in the future). 

#### Example Workflows

Examples that MadLib can help create for the matching step: 
* A *featurizing workflow* that creates features then uses them to convert all tuple pairs in the candidate set (which is the output of the blocking step) to a set of feature vectors.
* A *labeling workflow* to label a set of tuple pairs as match/non-match.
* A *training-data-creation workflow* that examines a very large set of tuple pairs to select a small set of tuple pairs that are "informative", then helps the user label this set. This workflow uses a well-known machine learning technique called active learning.
* A *sampling workflow* that outputs a sample of tuple pairs from a large set of tuple pairs. The sample is likely to contain both matches and non-matches.  
* A *matching-using-passive-learning workflow* that reads two tables A and B, a candidate set C of tuple pairs (obtained from running a blocking solution for A and B), a set of labeled tuple pairs P, featurizes C and P, trains a matcher M on P, then applies M to predict each pair in C as match/non-match. This workflow is well suited for the case where the user already has a set of labeled tuple pairs P to serve as training data for the matcher. 
* A *matching-using-active-learning* workflow that reads two tables A and B, a candidate set C of tuple pairs (obtained from running a blocking solution for A and B), takes a sample S of C, performs active learning on S to label a set of tuple pairs, uses this set to train a matcher M, and applies M to predict match/non-match for each tuple pair in C. This workflow is well suited for the case where the user does not yet have any training data for the matcher. 
* And many more possible workflows.

For example, MadLib functions can be combined to create a matching-using-active-learning workflow that is equivalent to the workflow used by [ActiveMatcher](https://github.com/anhaidgroup/active_matcher). See the Python script for this workflow [here](https://github.com/MadMatcher/MadLib/blob/main/examples/spark-cluster-examples/madlib_spark_cluster.py). 

### Runtime Environments

The above workflows are created by stitching together MadLib functions and Python code in Python scripts. You can run these scripts in three runtime environments: 
* *Pandas on a single machine:* Use this if you have a relatively small amount of data, or just want to experiment with MadLib.
* *Spark on a single machine:* Use this if you have a relatively small amount of data, or just want to experiment with MadLib, or if you want to test your Spark scripts before running them on a cluster.
*  *Spark on a cluster of machines:* Use this if you have a large amount of data, such as 5M+ tuples or more in Tables A and B. 

### Solving Matching Challenges

Today the matching step of EM often uses machine learning: train a matcher M on a set of labeled tuple pairs, then apply M to new pairs to predict match/non-match. In the EM context this raises a set of challenges, as described below. MadLib functions address these challenges, and distinguish MadLib from other existing EM packages: 
* *How to create the features?* MadLib analyzes the schema and data of the two tables A and B to be matched, to create a comprehensive set of features that involve similarity functions and tokenizers.
* *How to create the training data?* If a set of labeled tuple pairs for training is not available (a very common scenario), MadLib can help the user create such a set, using active learning.
* *How to scale?* When the tables A and B to be matched are large (e.g., 5M+ tuples), scaling is difficult. MadLib provides Spark-based solutions to these problems.
* *How to label examples?* MadLib provides a variety of labelers that the user can use (to label data in the command-line interface, using a Web browser, etc.), and ways to extend or customize these labelers.

=======================================





 

## Features

- **Machine Learning Models**: Support for both Scikit-Learn and PySpark ML models
- **Active Learning**: Efficient labeling with batch and continuous modes
- **Similarity Functions**: TF-IDF, Jaccard, Overlap, SIF embeddings, and Cosine Similarity
- **Flexible Labeling**: CLI-based, Web-based, and Gold-based labeling interfaces
- **Scalable Processing**: Can leverage PySpark for handling large datasets

## Installation

Please reference the installation guides. 

For single machine installation with Pandas, see the [single-machine installation guide](https://github.com/MadMatcher/MadLib/blob/main/docs/installation-guides/install-single-machine.md). 

For single machine installation with Spark, see the [single-machine installation guide](https://github.com/MadMatcher/MadLib/blob/main/docs/installation-guides/install-single-machine.md). 

For cloud-based cluster installation with Spark, see the [cloud-based-cluster installation guide](https://github.com/MadMatcher/MadLib/blob/main/docs/installation-guides/install-cloud-based-cluster.md).

## Quick Start

Here's a simple example of using MadLib:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from MadLib import create_features, featurize, create_seeds, train_matcher, GoldLabeler, SKLearnModel

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
    'id2': [101, 102, 103],
    'id1_list': [[1], [2], [3]]
})

# Create features for comparison
features = create_features(df_a, df_b, ['name', 'email'], ['name', 'email'])

# Generate feature vectors
fvs = featurize(features, df_a, df_b, candidates)

# Create labeled seeds for training
gold_df = pd.DataFrame({'id1': [1, 3], 'id2': [101, 103]})
gold_labeler = GoldLabeler(gold=gold_df)
seeds = create_seeds(fvs, nseeds=2, labeler=gold_labeler)

# Train a matcher model
model = SKLearnModel(
     model=LogisticRegression,
     nan_fill=0.0,
     random_state=42
)
trained_model = train_matcher(model, seeds)

# Apply the model to make predictions
predictions = apply_matcher(trained_model, fvs, 'feature_vectors', 'prediction', 'confidence')
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
- **`featurize(features, A, B, candidates, output_col='feature_vectors', fill_na=0.0)`**: Apply features to candidate pairs and obtain feature vectors
- **`down_sample(fvs, percent, search_id_column, score_column='score', bucket_size=1000)`**: Reduce dataset size by intelligently sampling to get matches and non-matches
- **`create_seeds(fvs, nseeds, labeler, score_column='score')`**: Create labeled training examples
- **`train_matcher(model_spec, labeled_data, feature_col='feature_vectors', label_col='label')`**: Train a matching model
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

Check out the comprehensive [Python notebook](https://github.com/MadMatcher/MadLib/blob/main/examples/madlib_examples.ipynb) for detailed usage examples including:

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

For an overview of the API, see the [API Docs](https://madmatcher.github.io/MadLib/).

For in-depth API documentation and explanations, see the [MadLib Technical Guide](https://github.com/MadMatcher/MadLib/blob/main/docs/MadLib-Technical-Guide.md).

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Support

For issues, questions, or feature requests, please:

1. Check the documentation and examples
2. Email us at entitymatchinginfo@gmail.com
