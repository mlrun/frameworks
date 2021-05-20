from typing import Union, List, Generator, Dict

from tensorflow.python.data import Dataset
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import Callback

from mlrun.execution import MLClientCtx
from frameworks._common.training.trainer import Trainer


class KerasTrainer(Trainer):
    """
    An interface for a tensorflow.keras model trainer, supporting the package's loggers and automatic logging.
    """

    def __init__(
        self,
        model: Model,
        training_set: Union[Sequence, Dataset, Generator],
        loss_function: Union[Loss, str],
        optimizer: Union[Optimizer, str],
        validation_set: Union[Sequence, Dataset, Generator] = None,
        metric_functions: List[Union[Metric, str]] = None,
        shuffle: Union[bool, str] = True,
        class_weight: Dict[int, float] = None,
        epochs: int = 1,
        initial_epoch: int = 0,
        training_iterations: int = None,
        validation_iterations: int = None,
        validation_frequency: Union[int, List[int]] = 1,
    ):
        """
        Initialize a trainer for a given experiment objects.
        :param model:                 The model to train.
        :param training_set:          The data for the training process. Can be passed as a keras Sequence, tf Dataset
                                      or a python Generator.
        :param loss_function:         The loss function to use during training. Can be passed as a string of one of the
                                      loss functions in 'tensorflow.keras.losses'.
        :param optimizer:             The optimizer to use during the training. Can be passed as a string of one of the
                                      optimizers in 'tensorflow.keras.optimizers'.
        :param validation_set:        The data for the validation process. Can be passed as a keras Sequence, tf Dataset
                                      or a python Generator.
        :param metric_functions:      The metrics to use on training and validation. Can be passed as a list of metrics
                                      or strings of the metrics in 'tensorflow.keras.metrics'.
        :param shuffle:               Whether or not to shuffle the training set given. Can be passed as a string
                                      equaled to 'batch' for shuffling per batch to deal with the limitations of HDF5
                                      data.
        :param class_weight:          A dictionary for telling the model to "pay more attention" to samples from an
                                      under-represented class. Expecting a dictionary where the keys are the classes
                                      indecis and the values are their weights.
        :param epochs:                Amount of epochs to perform. Defaulted to a single epoch.
        :param initial_epoch:         From which epoch to begin the training. Useful for resuming a previous training
                                      run.
        :param training_iterations:   Amount of iterations (batches) to perform on each epoch's training. If 'None' the
                                      entire training set will be used.
        :param validation_iterations: Amount of iterations (batches) to perform on each epoch's validation. If 'None'
                                      the entire validation set will be used.
        :param validation_frequency:  Per which amount ot epochs to run validation. Can be passed as a list of integers
                                      representing the epochs to run validation at.
        """
        # Store the model and datasets:
        self._model = model
        self._training_set = training_set
        self._validation_set = validation_set

        # Store the compile arguments:
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._metric_functions = (
            metric_functions if metric_functions is not None else []
        )

        # Store the run configuration:
        self._shuffle = shuffle
        self._class_weight = class_weight
        self._epochs = epochs
        self._initial_epoch = initial_epoch
        self._training_iterations = (
            len(training_set)
            if training_iterations is None or training_iterations > len(training_set)
            else training_iterations
        )
        self._validation_iterations = (
            len(validation_set)
            if validation_iterations is None
            or validation_iterations > len(validation_set)
            else validation_iterations
        )
        self._validation_frequency = validation_frequency

    def run(self, callbacks: List[Callback] = None):
        """
        Run the trainer training process on his initialized configuration.
        :param callbacks: The loggers to use on this run.
        """
        # Compile the model:
        # TODO: Apply missing arguments from the 'compile' method.
        self._model.compile(
            optimizer=self._optimizer,
            loss=self._loss_function,
            metrics=self._metric_functions,
        )

        # Initiate training:
        self._model.fit(
            x=self._training_set,
            epochs=self._epochs,
            verbose=2,
            callbacks=callbacks,
            validation_data=self._validation_set,
            shuffle=self._shuffle,
            class_weight=self._class_weight,
            initial_epoch=self._initial_epoch,
            steps_per_epoch=self._training_iterations,
            validation_steps=self._validation_iterations,
            validation_freq=self._validation_frequency,
        )

    def auto_log(self, context: MLClientCtx):
        """
        Run training with automatic logging to mlrun's context and tensorboard.
        :param context: The context to use for the logs.
        """
        raise NotImplementedError
