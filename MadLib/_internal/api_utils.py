from typing import Union, Dict
from .ml_model import MLModel, SKLearnModel, SparkMLModel
from .labeler import Labeler, CLILabeler, GoldLabeler, WebUILabeler
from pyspark.ml import Transformer
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError



AVAILABLE_LABELERS = {
    'cli': {
        'class': CLILabeler,
        'description': 'Command-line interactive labeler',
        'required_args': ['a_df', 'b_df'],
        'optional_args': ['id_col']
    },
    'gold': {
        'class': GoldLabeler,
        'description': 'Labeler that uses a gold standard set of matches',
        'required_args': ['gold'],
        'optional_args': []
    },
    'webui': {
        'class': WebUILabeler,
        'description': 'Web-based interactive labeler with Streamlit UI',
        'required_args': ['a_df', 'b_df'],
        'optional_args': ['id_col', 'flask_port', 'streamlit_port', 'flask_host']
    }
}

def _create_training_model(model_spec: Union[Dict, MLModel]) -> MLModel:
    # if the user passes in an instantiated MLModel, we don't need to create a new object
    if isinstance(model_spec, MLModel):
        return model_spec
    
    # if the user passes in a Dict, we need the following:
    # type (SKLearn vs SparkML, model class, args for the model itself)
    model_type = model_spec['model_type']
    model = model_spec['model']
    if model_type.lower() == 'sklearn':
        if 'execution' in model_spec:
            execution = model_spec['execution']
        else:
            execution = 'local'
        if 'nan_fill' in model_spec:
            nan_fill = model_spec['nan_fill']
        else:
            nan_fill = None
        if 'use_floats' in model_spec:
            use_floats = model_spec['use_floats']
        else:
            use_floats = True
        return SKLearnModel(model=model, nan_fill=nan_fill, use_floats=use_floats, execution=execution, **model_spec['model_args'])
    else:
        if 'nan_fill' in model_spec:
            nan_fill = model_spec['nan_fill']
        else:
            nan_fill = None
        return SparkMLModel(model=model, nan_fill=nan_fill, **model_spec['model_args'])


def _create_matching_model(model) -> MLModel:
    if isinstance(model, MLModel):
        if model.trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        else:
            return model
    if isinstance(model, Transformer):
        return SparkMLModel(model)
    try:
        check_is_fitted(model)
    except NotFittedError:
        raise RuntimeError('Model must be trained to predict')
    return SKLearnModel(model)


def _create_labeler(labeler_spec: Union[Dict, Labeler]) -> Labeler:
    if isinstance(labeler_spec, Labeler):
        return labeler_spec

    if 'name' not in labeler_spec:
        raise ValueError("Missing required key 'name' for labeler_spec")
    name = labeler_spec['name']
    info = AVAILABLE_LABELERS.get(name)
    if info is None:
        raise ValueError(f"Unknown labeler type '{name}'")
    init_kwargs = {}
    for arg in info['required_args']:
        if arg not in labeler_spec:
            raise ValueError(f"Missing required argument '{arg}' for labeler type '{name}'")
        init_kwargs[arg] = labeler_spec[arg]

    for arg in info['optional_args']:
        if arg in labeler_spec:
            init_kwargs[arg] = labeler_spec[arg]

    cls = info['class']
    return cls(**init_kwargs)
