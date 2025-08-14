"""
This workflow runs Spark on a single machine. It shows how you can use MadLib functions, 
especially labeled_pairs() to label a set of tuple pairs.  It loads the data, then asks 
the user to label the tuple pairs in the candidate set, using the CLI labeler (or the Web labeler). 
Note that it is straightforward to modify this script to take a sample from the candidate set 
using down_sample(), then asks the user to label only the pairs in the sample. 
"""

import warnings
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
# Import MadLib functions
from MadLib import CLILabeler, WebUILabeler, save_dataframe, load_dataframe, label_pairs
warnings.filterwarnings('ignore')

spark = SparkSession.builder \
   .master("local[*]") \
   .appName("MadLib Spark Labeling Only") \
   .config('spark.sql.execution.arrow.pyspark.enabled', 'true')\
   .getOrCreate()

# Load data
table_a = spark.read.parquet('../data/dblp_acm/table_a.parquet')
table_b = spark.read.parquet('../data/dblp_acm/table_b.parquet')
candidates = spark.read.parquet('../data/dblp_acm/cand.parquet')
candidates = candidates.withColumnRenamed('_id', 'id2').withColumnRenamed('ids', 'id1_list')
# explode the id1_list column to get id2, id1 pairs
candidates = candidates.select('id2', F.explode('id1_list').alias('id1'))

# Create CLI labeler
labeler = CLILabeler(
    a_df=table_a,
    b_df=table_b,
    id_col='_id'
)

"""
# Or, uncomment this block to create Web UI labeler
labeler = WebUILabeler(
    a_df=table_a,
    b_df=table_b,
    id_col='_id'
)
"""

labeled_pairs = label_pairs(
    labeler=labeler,
    pairs=candidates
)

# Save the labeled pairs to a parquet file
save_dataframe(labeled_pairs, 'labeled_pairs.parquet')

# to load the labeled pairs back in:
labeled_pairs = load_dataframe('labeled_pairs.parquet', 'sparkdf')
labeled_pairs.show()
