from typing import Union, Dict
from .ml_model import MLModel, SKLearnModel, SparkMLModel
from .labeler import Labeler, CLILabeler, GoldLabeler
from sklearn.base import BaseEstimator

AVAILABLE_MODELS = {
    'sklearn': {
        'class': SKLearnModel,
        'description': 'Scikit-learn model wrapper',
        'required_args': ['model_class'],
        'optional_args': ['**model_args']
    },
    'spark': {
        'class': SparkMLModel,
        'description': 'Spark ML model wrapper',
        'required_args': ['model'],
        'optional_args': ['nan_fill=0.0', '**model_args']
    }
}

AVAILABLE_LABELERS = {
    'cli': {
        'class': CLILabeler,
        'description': 'Command-line interactive labeler',
        'required_args': ['a_df', 'b_df'],
        'optional_args': ['id_col="_id"']
    },
    'gold': {
        'class': GoldLabeler,
        'description': 'Labeler that uses a gold standard set of matches',
        'required_args': ['gold'],
        'optional_args': []
    }
}

def _create_model(model_spec: Union[str, Dict, MLModel, BaseEstimator], **kwargs) -> MLModel:
    # 1) Shortcut if already wrapped or sklearn/spark estimator
    if isinstance(model_spec, MLModel):
        return model_spec
    if hasattr(model_spec, 'predict') or hasattr(model_spec, 'transform'):
        if hasattr(model_spec, 'transform'):
            return SparkMLModel(model_spec)
        return SKLearnModel(model_spec)

    # 2) Build a fresh config dict
    if isinstance(model_spec, str):
        config = {'name': model_spec, **kwargs}
    else:
        config = {**model_spec, **kwargs}  # shallow copy

    # 3) Extract “name” without pop side-effects
    if 'name' not in config:
        raise ValueError("Missing required key 'name' for model_spec")
    model_name = config['name']
    info = AVAILABLE_MODELS.get(model_name)
    if info is None:
        raise ValueError(f"Unknown model type '{model_name}'")

    # 4) Gather required args, checking but not popping the dict
    init_kwargs = {}
    for arg in info['required_args']:
        if arg not in config:
            raise ValueError(f"Missing required argument '{arg}' for model type '{model_name}'")
        init_kwargs[arg] = config[arg]

    # 5) Everything else in config (minus 'name') becomes optional args
    optional_kwargs = {k: v for k, v in config.items() if k not in {'name', *info['required_args']}}

    # 6) Finally instantiate
    cls = info['class']
    return cls(**init_kwargs, **optional_kwargs)


def _create_labeler(labeler_spec: Union[str, Dict, Labeler], **kwargs) -> Labeler:
    if isinstance(labeler_spec, Labeler):
        return labeler_spec

    if isinstance(labeler_spec, str):
        config = {'name': labeler_spec, **kwargs}
    else:
        config = {**labeler_spec, **kwargs}

    if 'name' not in config:
        raise ValueError("Missing required key 'name' for labeler_spec")
    name = config['name']
    info = AVAILABLE_LABELERS.get(name)
    if info is None:
        raise ValueError(f"Unknown labeler type '{name}'")

    init_kwargs = {}
    for arg in info['required_args']:
        if arg not in config:
            raise ValueError(f"Missing required argument '{arg}' for labeler type '{name}'")
        init_kwargs[arg] = config[arg]

    optional_kwargs = {k: v for k, v in config.items() if k not in {'name', *info['required_args']}}

    cls = info['class']
    return cls(**init_kwargs, **optional_kwargs)
