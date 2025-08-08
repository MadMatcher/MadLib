"""
This example shows how to use MadLib with pandas DataFrames.
Passive learning workflow using dblp_acm dataset with gold labeler.

The schema for the labeled pairs should be 'id2', 'id1_list', 'label'. 
We provide an example of how to create this format using labeled_pairs if your current schema is 'id2', 'id1', 'label'.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Import MadLib functions
from MadLib import (
    create_features, featurize, 
    train_matcher, apply_matcher, 
)
from MadLib import SKLearnModel

# Load data
table_a = pd.read_parquet('../data/dblp_acm/table_a.parquet')
table_b = pd.read_parquet('../data/dblp_acm/table_b.parquet')
candidates = pd.read_parquet('../data/dblp_acm/cand.parquet')
candidates = candidates.rename(columns={'_id': 'id2', 'ids': 'id1_list'})
candidates = candidates[['id2', 'id1_list']]

# Load and format labeled pairs
labeled_pairs = pd.read_parquet('../data/dblp_acm/gold.parquet')
# In this example, our gold data only contains matches. So, labeled pairs are all matches. 
# We need to add a negative example pair to the labeled pairs, or else the model will not learn to predict 0.0. 
# In a real-world setting, you would have labeled data that contains both matches and non-matches.
new_pair = pd.DataFrame({'id2': [0], 'id1': [1], 'label': [0.0]})
labeled_pairs = pd.concat([labeled_pairs, new_pair])
# Our labeled pairs are all matches , so we set the label to 1.0 for the original pairs
# (but preserve the 0.0 we just added for the new pair)
labeled_pairs.loc[labeled_pairs['label'] != 0, 'label'] = 1.0

# Our current schema is 'id2', 'id1', 'label'. We need to convert it to 'id2', 'id1_list', 'label' for featurization.
labeled_pairs = labeled_pairs.groupby('id2', as_index=False).agg({
    'id1': list,
    'label': list
})
labeled_pairs = labeled_pairs.rename(columns={'id1': 'id1_list'})


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

print(predictions.head())
