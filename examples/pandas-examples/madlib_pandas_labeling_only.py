"""
This workflow runs Pandas on a single machine. It shows how you can use MadLib functions, 
especially labeled_pairs() to label a set of tuple pairs.  It loads the data, then asks 
the user to label the tuple pairs in the candidate set, using the CLI labeler (or the Web labeler). 
Note that it is straightforward to modify this script to take a sample from the candidate set 
using down_sample(), then asks the user to label only the pairs in the sample. 
"""

import pandas as pd
import warnings
# Import MadLib functions
from MadLib import CLILabeler, WebUILabeler, save_dataframe, load_dataframe, label_pairs
warnings.filterwarnings('ignore')

# Load data
table_a = pd.read_parquet('../data/dblp_acm/table_a.parquet')
table_b = pd.read_parquet('../data/dblp_acm/table_b.parquet')
candidates = pd.read_parquet('../data/dblp_acm/cand.parquet')
candidates = candidates.rename(columns={'_id': 'id2', 'ids': 'id1_list'})

# Convert from id2: id1_list to id2: id1 pairs
candidates = candidates.explode('id1_list').rename(columns={'id1_list': 'id1'})
candidates = candidates[['id2', 'id1']]

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
labeled_pairs = load_dataframe('labeled_pairs.parquet', 'pandas')
print(labeled_pairs.head())
