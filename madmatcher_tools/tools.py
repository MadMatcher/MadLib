"""
Public API functions for MadMatcher.

This module provides the main functions that users will interact with.
Implementation details are hidden in the _internal package.
"""

from typing import List, Optional, Callable, Any, Union, Literal, Dict, Type
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator
import xxhash

from ._internal.ml_model import MLModel, SKLearnModel, SparkMLModel
from ._internal.labeler import Labeler, CLILabeler, GoldLabeler
from ._internal.active_learning.ent_active_learner import EntropyActiveLearner
from ._internal.active_learning.cont_entropy_active_learner import ContinuousEntropyActiveLearner
from ._internal.featurization import (
    create_features,
    get_base_sim_functions,
    get_base_tokenizers,
    get_extra_tokenizers,
    featurize
)
from ._internal.api_utils import _create_model, _create_labeler

# Re-export the public functions
__all__ = [
    'create_features',
    'get_base_sim_functions',
    'get_base_tokenizers',
    'get_extra_tokenizers',
    'featurize',
    'down_sample',
    'select_seeds',
    'train_matcher',
    'apply_matcher',
    'label_data'
]

def down_sample(
    fvs: pd.DataFrame,
    percent: float,
    search_id_column: str,
    score_column: str = 'score',
    bucket_size: int = 1_000,
) -> pd.DataFrame:
    """
    down sample by score_column to produce percent * fvs.count() rows

    Parameters
    ----------
    fvs : pandas DataFrame
        the feature vectors to be downsampled
    percent : float
        the portion of the vectors to be output, (0.0, 1.0]
    score_column : str
        the column that scored the vectors, should be positively correlated with the probability of the pair being a match

    Returns
    -------
    pandas DataFrame
        the down sampled dataset with percent * fvs.count() rows with the same schema as fvs
    """
    if not 0 <= percent <= 1:
        raise ValueError("percent must be between 0 and 1")
    if percent == 0:
        return pd.DataFrame(columns=fvs.columns)
    if percent == 1:
        return fvs.copy()
    fvs = fvs.copy()
    total = len(fvs)
    nparts = max(total // bucket_size, 1)
    fvs["_hash_bucket"] = fvs[search_id_column].astype(str).apply(lambda val: xxhash.xxh64(val).intdigest() % nparts)
    fvs["_rank_desc"] = fvs.groupby("_hash_bucket")[score_column] \
        .rank(method="min", ascending=False)

    bucket_sizes = fvs["_hash_bucket"].value_counts().to_dict()
    fvs["_bucket_size"] = fvs["_hash_bucket"].map(bucket_sizes)

    fvs["_percent_rank"] = fvs.apply(
        lambda row: 0.0
        if row["_bucket_size"] == 1
        else (row["_rank_desc"] - 1) / (row["_bucket_size"] - 1),
        axis=1,
    )
    fvs = fvs[fvs["_percent_rank"] <= percent].copy()
    fvs = fvs.drop(columns=["_hash_bucket", "_rank_desc", "_bucket_size", "_percent_rank"])

    return fvs


def select_seeds(
    fvs: pd.DataFrame,
    nseeds: int,
    labeler: Union[Labeler, Dict],
    score_column: str = 'score'
) -> pd.DataFrame:
    """
    select seeds to train a model

    Parameters
    ----------
    fvs : pandas DataFrame
        the DataFrame with feature vectors that is your training data
    nseeds : int
        the number of seeds you want to use to train an initial model
    labeler : Union[Labeler, Dict]
        the labeler object (or a labeler_spec dict) you want to use to assign labels to rows
    score_column : str
        the name of the score column in your fvs DataFrame

    Returns
    -------
    pandas DataFrame
        A DataFrame with labeled seeds, schema is (previous schema of fvs, `label`) where the values in 
        label are either 0.0 or 1.0
    """
    if isinstance(labeler, dict):
        labeler = _create_labeler(labeler)
    if len(fvs) == 0 or nseeds == 0:
        df = fvs.copy()
        df['label'] = []
        return df
    fvs = fvs[fvs[score_column].notna()].copy()
    if len(fvs) == 0:
        df = fvs.copy()
        df['label'] = []
        return df
    maybe_pos = fvs.nlargest(nseeds, score_column).iterrows()
    maybe_neg = fvs.nsmallest(nseeds, score_column).iterrows()

    pos_count = 0
    neg_count = 0
    seeds = []
    i = 0
    while pos_count + neg_count < nseeds and i < nseeds * 2:
        try:
            _, ex = next(maybe_pos) if pos_count <= neg_count else next(maybe_neg)
            label = float(labeler(ex['id1'], ex['id2']))
            
            if label == -1.0:  # User requested to stop
                break
            elif label == 2.0:  # User marked as unsure
                continue
            elif label == 1.0:  # Positive match
                pos_count += 1
            else:  # label == 0.0, Negative match
                neg_count += 1

            ex['label'] = label
            seeds.append(ex)
        except StopIteration:
            print("Ran out of examples before reaching nseeds")
            break
        i += 1
    if not seeds:
        raise RuntimeError("No seeds were labeled before stopping")
    
    print(f"seeds: pos_count = {pos_count} neg_count = {neg_count}")
    return pd.DataFrame(seeds)


def train_matcher(
    model_spec: Union[str, Dict, MLModel, BaseEstimator],
    labeled_data: pd.DataFrame,
    feature_col: str = "features",
    label_col: str = "label",
    **model_kwargs
) -> MLModel:
    """Train a matcher model on labeled data.
    
    Parameters
    ----------
    model_spec : Union[str, Dict, MLModel, BaseEstimator]
        Either:
        - A string name of a built-in model (e.g. 'sklearn')
        - A dict with model configuration (e.g. {'name': 'sklearn', 'model_class': XGBClassifier, 'max_depth': 6})
        - An MLModel instance
        - A scikit-learn or Spark model instance
    labeled_data : pandas DataFrame
        DataFrame containing the labeled data
    feature_col : str, optional
        Name of the column containing feature vectors
    label_col : str, optional
        Name of the column containing labels
    **model_kwargs : dict
        Additional arguments to pass to the model constructor
        
    Returns
    -------
    MLModel
        The trained model
    """
    model = _create_model(model_spec, **model_kwargs)
    return model.train(labeled_data, feature_col, label_col)


def apply_matcher(
    model_spec: Union[str, Dict, MLModel, BaseEstimator],
    df: pd.DataFrame,
    feature_col: str,
    output_col: str,
    **model_kwargs
) -> pd.DataFrame:
    """Apply a trained model to make predictions.
    
    Parameters
    ----------
    model_spec : Union[str, Dict, MLModel, BaseEstimator]
        Either:
        - A string name of a built-in model (e.g. 'sklearn')
        - A dict with model configuration (e.g. {'name': 'sklearn', 'model_class': XGBClassifier})
        - An MLModel instance
        - A scikit-learn or Spark model instance
    df : pandas DataFrame
        The DataFrame to make predictions on
    feature_col : str
        Name of the column containing feature vectors
    output_col : str
        Name of the column to store predictions in
    **model_kwargs : dict
        Additional arguments to pass to the model constructor
        
    Returns
    -------
    pandas DataFrame
        The input DataFrame with predictions added
    """
    model = _create_model(model_spec, **model_kwargs)
    return model.predict(df, feature_col, output_col)


def label_data(
    model_spec: Union[str, Dict, MLModel, BaseEstimator],
    mode: Literal["batch", "continuous"],
    labeler_spec: Union[str, Dict, Labeler],
    fvs: pd.DataFrame,
    seeds: Optional[pd.DataFrame] = None,
    **kwargs
) -> pd.DataFrame:
    """Generate labeled data using active learning.
    
    Parameters
    ----------
    model_spec : Union[str, Dict, MLModel, BaseEstimator]
        Either:
        - A string name of a built-in model (e.g. 'sklearn')
        - A dict with model configuration (e.g. {'name': 'sklearn', 'model_class': XGBClassifier})
        - An MLModel instance
        - A scikit-learn or Spark model instance
    mode : Literal["batch", "continuous"]
        Whether to use batch or continuous active learning
    labeler_spec : Union[str, Dict, Labeler]
        Either:
        - A string name of a built-in labeler (e.g. 'cli')
        - A dict with labeler configuration (e.g. {'name': 'cli', 'a_df': df_a, 'b_df': df_b})
        - A Labeler instance
    fvs : pandas DataFrame
        The data that needs to be labeled
    seeds : pandas DataFrame, optional
        Initial labeled examples to start with
    **kwargs : dict
        Additional arguments to pass to the model or labeler constructors
        
    Returns
    -------
    pandas DataFrame
        DataFrame with ids of potential matches and the corresponding label
    """
    spark = SparkSession.builder.getOrCreate()
    
    # Create model and labeler
    model = _create_model(model_spec, **kwargs)
    labeler = _create_labeler(labeler_spec, **kwargs)
    
    if seeds is None:
        seeds = select_seeds(fvs=fvs, nseeds=10, labeler=labeler, score_column='score')
    
    if mode == "batch":
        learner = EntropyActiveLearner(model, labeler)
    elif mode == "continuous":
        learner = ContinuousEntropyActiveLearner(model, labeler)
    
    if isinstance(fvs, pd.DataFrame):
        fvs = spark.createDataFrame(fvs)
    
    labeled_data = learner.train(fvs, seeds)
    return labeled_data
