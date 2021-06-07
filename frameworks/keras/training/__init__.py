import tensorflow as tf
from tensorflow import keras

import mlrun

from frameworks.keras.training.mlrun_model import MLRunModel


def wrap_model(
    model: keras.Model,
    context: mlrun.MLClientCtx,
    auto_log: bool = True,
    use_horovod: bool = False,
) -> keras.Model:
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.

    :param model:       The model to wrap.
    :param context:     MLRun context to work with.
    :param auto_log:    Whether or not to apply MLRun's auto logging on the model. Defaulted to True.
    :param use_horovod: Whether or not to use horovod for training. Defaulted to False.

    :return: MLRun model wrapping the model.
    """
    model = MLRunModel.wrap(model=model)
    if use_horovod:
        model.use_horovod()
    if context is not None and auto_log:
        model.auto_log(context=context)
    return model
