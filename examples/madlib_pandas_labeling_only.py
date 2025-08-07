"""
This example shows how to use MadLib with pandas DataFrames.
Labeling only with the dblp_acm dataset.
"""

import pandas as pd
import warnings
# Import MadLib functions
from MadLib import CLILabeler, WebUILabeler, save_dataframe, load_dataframe
warnings.filterwarnings('ignore')

# Load data
table_a = pd.read_parquet('./data/dblp_acm/table_a.parquet')
table_b = pd.read_parquet('./data/dblp_acm/table_b.parquet')
candidates = pd.read_parquet('./data/dblp_acm/cand.parquet')
candidates = candidates.rename(columns={'_id': 'id2', 'ids': 'id1_list'})
candidates = candidates[['id2', 'id1_list']]
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

label = 0
labeled_candidates = []
for index, row in candidates.iterrows():
    id2 = row['id2']
    id1_list = row['id1_list']
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

labeled_candidates = pd.DataFrame(labeled_candidates)

# Save the labeled pairs to a parquet file
save_dataframe(labeled_candidates, 'labeled_candidates.parquet')

# to load the labeled candidates back in:
labeled_candidates = load_dataframe('labeled_candidates.parquet', 'pandas')
labeled_candidates.head()
