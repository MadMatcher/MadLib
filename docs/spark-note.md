## Notes on Spark

### Setting Up a SparkSession

Spark works by processing data in parallel, which significantly increases processing speed. On a local machine, Spark treats each CPU core as a worker node to distribute work efficiently.

Before using Spark DataFrames with MatchFlow, you need to set up a SparkSession:

```python
from pyspark.sql import SparkSession

# Create a SparkSession for local processing
spark = SparkSession.builder \
    .master('local[*]') \
    .appName('MatchFlow') \
    .getOrCreate()
```

**Configuration Options:**

- `master('local[*]')`: Uses all available CPU cores on your local machine
- `master('local[4]')`: Uses exactly 4 CPU cores
- `master(url)`: Uses the machine at URL as the driver
- `appName('MatchFlow')`: Names your Spark application for identification

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
from MatchFlow import create_features, featurize

# Set up Spark
spark = SparkSession.builder \
    .master('local[*]') \
    .appName('MatchFlow') \
    .getOrCreate()

# Convert your data to Spark DataFrames
spark_df_a = spark.createDataFrame(df_a)
spark_df_b = spark.createDataFrame(df_b)
spark_candidates = spark.createDataFrame(candidates)

# Use the same MatchFlow functions
features = create_features(spark_df_a, spark_df_b, ['name', 'email'], ['name', 'email'])
spark_fvs = featurize(features, spark_df_a, spark_df_b, spark_candidates)

# The result is a Spark DataFrame with distributed processing
print(f"Feature vectors count: {spark_fvs.count()}")
```

For additional SparkSession configuration options (storage limits, memory settings, etc.), refer to the [official PySpark documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.html).
