from typing import Union, List, Dict, Tuple
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.metrics import Metric
import mlrun
from mlrun import MLClientCtx
import horovod.tensorflow.keras as hvd
from frameworks.keras.utilities import KerasHorovodHandler
from frameworks.keras.callbacks import (
    MLRunLoggingCallback,
    TensorboardLoggingCallback,
    TrackableType,
)


def get_auto_logging_callbacks(
    context: MLClientCtx, static_hyperparameters: Dict[str, TrackableType] = None
) -> List[Callback]:
    """
    Get the defaulted logging callbacks by MLRun. Given the context, the method will setup a list of callbacks with the
    most common settings for logging a training session in tensorflow.keras. The returned callbacks list can be given
    to the 'compile_with_horovod' method for adding the callbacks to work with the horovod framework. For further
    information regarding the logging callbacks, see 'mlrun.frameworks.keras.callbacks.MLRunLoggingCallback' and
    'mlrun.frameworks.keras.callbacks.TensorboardLoggingCallback'.

    :param context:                The MLRun context to log with.
    :param static_hyperparameters: A dictionary of static hyperparameters to note in the logs.

    :return: The initialized logging callbacks of MLRun.
    """
    dynamic_hyperparameters = {"learning_rate": ["optimizer", "lr"]}
    return [
        MLRunLoggingCallback(
            context=context,
            static_hyperparameters=static_hyperparameters,
            dynamic_hyperparameters=dynamic_hyperparameters,
        ),
        TensorboardLoggingCallback(
            context=context,
            static_hyperparameters=static_hyperparameters,
            dynamic_hyperparameters=dynamic_hyperparameters,
            weights=True,
        ),
    ]


# List of all callbacks to should be applied only on rank 0 while using horovod:
_RANK_0_ONLY_CALLBACKS = [
    ModelCheckpoint.__name__,
    TensorBoard.__name__,
    MLRunLoggingCallback.__name__,
    TensorboardLoggingCallback.__name__,
]  # type: List[str]


def compile_with_horovod(
    model: Model,
    loss: Union[Loss, str],
    optimizer: Optimizer,
    metrics: List[Union[Metric, str]] = None,
    loss_weights: Union[List[float], Dict[str, float]] = None,
    weighted_metrics: List[Union[Metric, str]] = None,
    run_eagerly: bool = None,
    steps_per_execution: int = None,
    steps_per_epoch: int = None,
    validation_steps: int = None,
    callbacks: List[Callback] = None,
    use_cuda: bool = True,
) -> Tuple[Model, Optimizer, dict]:
    """
    Compile the given model for running with horovod - a distributed training framework. The returned tuple will have:

    * The given model compiled with the given parameters (see tensorflow.keras.Model.compile() for better reference of
      the compile method).
    * The given optimizer wrapped with horovod's distributed optimizer.
    * The adjusted attributes in a key word dictionary ready for tensorflow.keras.Model.fit().

    :param model:               The model to compile.
    :param loss:                Loss function as in 'tensorflow.keras.Model.compile()'.
    :param optimizer:           An optimizer instance. Note that unlike in 'tf.keras.Model.compile()', the optimizer
                                must be an initialized instance of an optimizer and it cannot be passed as a string.
    :param metrics:             List of metric functions as in 'tf.keras.Model.compile()'.
    :param loss_weights:        A list of scalars to apply on the outputs or a dictionary of output names to scalars as
                                in 'tf.keras.Model.compile()'.
    :param weighted_metrics:    List of metric functions as in 'tf.keras.Model.compile()'.
    :param run_eagerly:         Whether or not to wrap the model in 'tf.function' as in 'tf.keras.Model.compile()'.
    :param steps_per_execution: Number of steps to run on each 'tf.function' call as in 'tf.keras.Model.compile()'.
    :param steps_per_epoch:     Number of training steps to run on each epoch as passed to 'tf.keras.Model.fit()'.
    :param validation_steps:    Number of validation steps to run on each epoch as passed to 'tf.keras.Model.fit()'.
    :param callbacks:           A list of callbacks to join with horovod callbacks and to filter by horovod's rank.
    :param use_cuda:            Whether or not to use cuda (if available) with horovod. True will use cuda and False
                                will not. Defaulted to True.

    :return: A tuple of:
             [0] = Horovod compiled model
             [1] = Wrapped optimizer
             [2] = A key word arguments for the 'tensorflow.keras.Model.fit()' method with horovod adjusted arguments:
                   * Adjusted training steps per epoch or None if not given.
                   * Adjusted validation steps per epoch or None if not given.
                   * Horovod applied callbacks list.
                   * Verbose value.
    """
    # # Import horovod:
    # KerasHorovodHandler.import_horovod()
    # hvd = KerasHorovodHandler.get_horovod()

    # Initialize horovod:
    hvd.init()

    # Setup the device to run on"
    if use_cuda and tf.config.experimental.list_physical_devices("GPU"):
        # Pin each GPU to a single process:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    else:
        # No GPUs were found, or 'use_cuda' was false:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Adjust the learning rate based on the number of GPUs:
    optimizer.lr = optimizer.lr * hvd.size()

    # Wrap the optimizer in horovod's distributed optimizer: in hvd.DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer)

    # Compile the model with `experimental_run_tf_function=False` to ensure Tensorflow uses the distributed optimizer
    # to compute gradients:
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        experimental_run_tf_function=False,
        loss_weights=loss_weights,
        weighted_metrics=weighted_metrics,
        run_eagerly=run_eagerly,
        steps_per_execution=steps_per_execution,
    )

    # Setup the callbacks:
    callbacks = [] if callbacks is None else callbacks
    horovod_callbacks = [
        # Broadcast initial variable states from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        # Average metrics among workers at the end of every epoch. This callback must be in the list before the
        # 'ReduceLROnPlateau', TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
        # Using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final accuracy. Scale the learning rate
        # `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during the first five epochs. See https://arxiv.org/abs/1706.02677
        # for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
    ]
    if hvd.rank() != 0:
        callbacks = [
            callback
            for callback in callbacks
            if callback.__name__ not in _RANK_0_ONLY_CALLBACKS
        ]
    callbacks = horovod_callbacks + callbacks

    # Pick the verbose:
    verbose = 1 if hvd.rank() == 0 else 0

    # Prepare the fit key word arguments dictionary:
    fit_kwargs = {"callbacks": callbacks, "verbose": verbose}

    # Adjust the number of steps per epoch based on the number of GPUs (if given):
    if steps_per_epoch is not None:
        fit_kwargs["steps_per_epoch"] = steps_per_epoch // hvd.size()
    if validation_steps is not None:
        fit_kwargs["validation_steps"] = validation_steps // hvd.size()

    return model, optimizer, fit_kwargs
