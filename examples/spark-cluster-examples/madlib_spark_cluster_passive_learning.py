"""
This example shows how to use MadLib with Spark DataFrames on a cluster.
Passive learning workflow using dblp_acm dataset with gold labeler.

The schema for the labeled pairs should be 'id2', 'id1_list', 'label'. 
'label' is a list of labels for each id1 in the id1_list.
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
labeled_pairs = spark.read.parquet(str(data_dir / 'labeled_pairs.parquet'))


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
