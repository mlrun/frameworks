import frameworks.keras.callbacks

from tensorflow import keras

import mlrun
from frameworks.keras.model_handler import KerasModelHandler
from frameworks.keras.mlrun_interface import KerasMLRunInterface


def apply_mlrun(
    model: keras.Model,
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    use_horovod: bool = False,
) -> keras.Model:
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.

    :param model:       The model to wrap.
    :param context:     MLRun context to work with. If no context is given it will be retrieved via
                        'mlrun.get_or_create_ctx(None)'
    :param auto_log:    Whether or not to apply MLRun's auto logging on the model. Defaulted to True.
    :param use_horovod: Whether or not to use horovod for training. Defaulted to False.

    :return: MLRun model wrapping the model.
    """
    if context is None:
        context = mlrun.get_or_create_ctx(None)
    KerasMLRunInterface.add_interface(model=model)
    if use_horovod:
        model.use_horovod()
    if context is not None and auto_log:
        model.auto_log(context=context)
    return model
