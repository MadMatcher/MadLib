"""
This example shows how to use MadLib with Spark DataFrames on a cluster.
Labeling only with the dblp_acm dataset.
"""

import warnings
from pyspark.sql import SparkSession
from pathlib import Path
import pyspark.sql.functions as F
# Import MadLib functions
from MadLib import WebUILabeler, save_dataframe, load_dataframe, label_pairs
warnings.filterwarnings('ignore')

spark = SparkSession.builder \
   .master("{url of spark master node}") \
   .appName("MadLib Spark Labeling Only") \
   .config('spark.sql.execution.arrow.pyspark.enabled', 'true')\
   .getOrCreate()

# Load data
data_dir = Path(__file__).resolve().parent.parent / 'data' / 'dblp_acm'
table_a = spark.read.parquet(str(data_dir / 'table_a.parquet'))
table_b = spark.read.parquet(str(data_dir / 'table_b.parquet'))
candidates = spark.read.parquet(str(data_dir / 'cand.parquet'))
candidates = candidates.withColumnRenamed('_id', 'id2').withColumnRenamed('ids', 'id1_list')
# explode the id1_list column to get id2, id1 pairs
candidates = candidates.select('id2', F.explode('id1_list').alias('id1'))

# with default settings, you can open the web UI labeler by going to:
# http://{public ip of spark master node}:8501
# from your local machine's browser
labeler = WebUILabeler(
    a_df=table_a,
    b_df=table_b,
    id_col='_id'
)

labeled_pairs = label_pairs(
    labeler=labeler,
    pairs=candidates
)

labeled_pairs = spark.createDataFrame(labeled_pairs)

# Save the labeled pairs to a parquet file on the master node
save_path = str(data_dir / 'labeled_pairs.parquet')
save_dataframe(labeled_pairs, save_path)

# to load the labeled pairs back in from the master node:
labeled_pairs = load_dataframe(save_path, 'sparkdf')
labeled_pairs.show()
