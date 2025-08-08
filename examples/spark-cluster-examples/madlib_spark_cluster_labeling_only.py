"""
This example shows how to use MadLib with Spark DataFrames on a cluster.
Labeling only with the dblp_acm dataset.
"""

import warnings
from pyspark.sql import SparkSession
from pathlib import Path
# Import MadLib functions
from MadLib import WebUILabeler, save_dataframe, load_dataframe
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
candidates = candidates.select('id2', 'id1_list')

# with default settings, you can open the web UI labeler by going to:
# http://{public ip of spark master node}:8501
# from your local machine's browser
labeler = WebUILabeler(
    a_df=table_a,
    b_df=table_b,
    id_col='_id'
)

label = 0
labeled_candidates = []
for row in candidates.collect():
    id2 = row.id2
    id1_list = row.id1_list
    for id1 in id1_list:
        label = labeler(id1, id2)
        if label == -1.0:   # -1.0 means the user wants to stop labeling
            break
        labeled_candidates.append({
            'id1': id1,
            'id2': id2,
            'label': label
        })
    if label == -1.0:   # -1.0 means the user wants to stop labeling
        break

labeled_candidates = spark.createDataFrame(labeled_candidates)

# Save the labeled pairs to a parquet file on the master node
save_path = str(data_dir / 'labeled_candidates.parquet')
save_dataframe(labeled_candidates, save_path)

# to load the labeled candidates back in from the master node:
labeled_candidates = load_dataframe(save_path, 'sparkdf')
labeled_candidates.show()
