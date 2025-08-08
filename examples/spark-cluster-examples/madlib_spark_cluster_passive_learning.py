"""
This example shows how to use MadLib with Spark DataFrames on a cluster.
Passive learning workflow using dblp_acm dataset with gold labeler.

The schema for the labeled pairs should be 'id2', 'id1_list', 'label'. 
We provide an example of how to create this format using labeled_pairs if your current schema is 'id2', 'id1', 'label'.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, collect_list
from xgboost import XGBClassifier
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import MadLib functions
from MadLib import (
    create_features, featurize, 
    train_matcher, apply_matcher, 
)
from MadLib import SKLearnModel

# Initialize Spark session
spark = SparkSession.builder \
    .master("{url of spark master node}") \
    .appName("MadLib Spark Passive Learning Example") \
    .config('spark.sql.execution.arrow.pyspark.enabled', 'true') \
    .getOrCreate()

# Load data using Spark
data_dir = Path(__file__).resolve().parent.parent / 'data' / 'dblp_acm'
table_a = spark.read.parquet(str(data_dir / 'table_a.parquet'))
table_b = spark.read.parquet(str(data_dir / 'table_b.parquet'))
candidates = spark.read.parquet(str(data_dir / 'cand.parquet'))
candidates = candidates.withColumnRenamed('_id', 'id2') \
    .withColumnRenamed('ids', 'id1_list') \
    .select('id2', 'id1_list')

# Load and format labeled pairs
labeled_pairs = spark.read.parquet(str(data_dir / 'gold.parquet'))

# Select only the columns we need (id1 and id2) and add label column
# In this example, our gold data only contains matches. So, labeled pairs are all matches (label = 1.0) 
labeled_pairs = labeled_pairs.select('id1', 'id2').withColumn('label', lit(1.0))

# We need to add a negative example pair to the labeled pairs, or else the model will not learn to predict 0.0. 
# In a real-world setting, you would have labeled data that contains both matches and non-matches.
# Add a negative example pair
new_pair = spark.createDataFrame([(1, 0, 0.0)], ['id1', 'id2', 'label'])
labeled_pairs = labeled_pairs.union(new_pair)

# Our current schema is 'id2', 'id1', 'label'. We need to convert it to 
# 'id2', 'id1_list', 'label' for featurization.
labeled_pairs = labeled_pairs.groupBy('id2').agg(
    collect_list('id1').alias('id1_list'),
    collect_list('label').alias('label')
)

# Create features
features = create_features(
    A=table_a,
    B=table_b,
    a_cols=['title', 'authors', 'venue', 'year'],
    b_cols=['title', 'authors', 'venue', 'year']
)

# Featurize labeled pairs
labeled_pairs_feature_vectors = featurize(
    features=features,
    A=table_a,
    B=table_b,
    candidates=labeled_pairs,  # Use grouped data with id1_list
    output_col='feature_vectors',
    fill_na=0.0
)

# Featurize candidate set
candidate_feature_vectors = featurize(
    features=features,
    A=table_a,
    B=table_b,
    candidates=candidates,
    output_col='feature_vectors',
    fill_na=0.0
)

# Create ML model
model = SKLearnModel(
    model=XGBClassifier,
    eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42,
    nan_fill=0.0
)

# Train matcher
trained_model = train_matcher(
    model=model,
    labeled_data=labeled_pairs_feature_vectors,
    feature_col='feature_vectors',
    label_col='label'
)

# Apply matcher
predictions = apply_matcher(
    model=trained_model,
    df=candidate_feature_vectors,
    feature_col='feature_vectors',
    prediction_col='prediction',
    confidence_col='confidence'
)

predictions.show()
