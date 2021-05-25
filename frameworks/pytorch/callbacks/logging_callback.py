from typing import Union, List, Dict, Tuple
import numpy as np
from torch import Tensor
from torch.nn import Module, Parameter
from frameworks._common.loggers import Logger, TrackableType
from frameworks.pytorch.callbacks.callback import (
    Callback,
    MetricValueType,
    MetricFunctionType,
)


class HyperparametersKeys:
    """
    For easy noting on which object to search for the hyperparameter to track with the logging callback in its
    initialization method.
    """

    MODEL = "model"
    TRAINING_SET = "training_set"
    VALIDATION_SET = "validation_set"
    LOSS_FUNCTION = "loss_function"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"


class LoggingCallback(Callback):
    """
    Callback for collecting data during training / evaluation. All the collected data will be available in this callback
    post the training / validation process and can be accessed via the 'training_results', 'validation_results',
    'static_hyperparameters', 'dynamic_hyperparameters' and 'summaries' properties.
    """

    class _MetricType:
        """
        Metric can be of two types, a loss metric or accuracy metric.
        """

        LOSS = "Loss"
        ACCURACY = "Accuracy"

    def __init__(
        self,
        dynamic_hyperparameters: Dict[str, Tuple[str, List[Union[str, int]]]] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, Tuple[str, List[Union[str, int]]]]
        ] = None,
        per_iteration_logging: int = 1,
    ):
        """
        Initialize a logging callback with the given hyperparameters and logging configurations.

        :param dynamic_hyperparameters: If needed to track a hyperparameter dynamically (sample it each epoch) it should
                                        be passed here. The parameter expects a dictionary where the keys are the
                                        hyperparameter chosen names and the values are tuples of object key and a list
                                        with the key chain. A key chain is a list of keys and indices to know how to
                                        access the needed hyperparameter. For example, to track the 'lr' attribute of
                                        an optimizer, one should pass:
                                        {
                                            "learning rate": (HyperparametersKeys.OPTIMIZER, ["param_groups", 0, "lr"])
                                        }
        :param static_hyperparameters:  If needed to track a hyperparameter one time per run it should be passed here.
                                        The parameter expects a dictionary where the keys are the
                                        hyperparameter chosen names and the values are the hyperparameter static value
                                        or a tuple of object key and a list with the key chain just like the dynamic
                                        hyperparameter. For example, to track the 'epochs' of an experiment run, one
                                        should pass:
                                        {
                                            "epochs": 7
                                        }
        :param per_iteration_logging:   Per how many iterations (batches) the callback should log the tracked values.
                                        Defaulted to 1 (meaning every iteration will be logged).
        """
        super(LoggingCallback, self).__init__()

        # Store the configurations:
        self._per_iteration_logging = per_iteration_logging
        self._dynamic_hyperparameters_keys = dynamic_hyperparameters
        self._static_hyperparameters_keys = static_hyperparameters

        # Initialize the logger:
        self._logger = Logger()

        # Setup the logger flag:
        self._log_iteration = False

    def get_training_results(self) -> Dict[str, List[List[float]]]:
        """
        Get the training results logged. The results will be stored in a dictionary where each key is the metric name
        and the value is a list of lists of values. The first list is by epoch and the second list is by iteration
        (batch).

        :return: The training results.
        """
        return self._logger.training_results

    def get_validation_results(self) -> Dict[str, List[List[float]]]:
        """
        Get the validation results logged. The results will be stored in a dictionary where each key is the metric name
        and the value is a list of lists of values. The first list is by epoch and the second list is by iteration
        (batch).

        :return: The validation results.
        """
        return self._logger.validation_results

    def get_static_hyperparameters(self) -> Dict[str, TrackableType]:
        """
        Get the static hyperparameters logged. The hyperparameters will be stored in a dictionary where each key is the
        hyperparameter name and the value is his logged value.

        :return: The static hyperparameters.
        """
        return self._logger.static_hyperparameters

    def get_dynamic_hyperparameters(self) -> Dict[str, List[TrackableType]]:
        """
        Get the dynamic hyperparameters logged. The hyperparameters will be stored in a dictionary where each key is the
        hyperparameter name and the value is a list of his logged values per epoch.

        :return: The dynamic hyperparameters.
        """
        return self._logger.dynamic_hyperparameters

    def get_summaries(self) -> Dict[str, List[float]]:
        """
        Get the validation summaries of the metrics results. The summaries will be stored in a dictionary where each key
        is the metric names and the value is a list of all the summary values per epoch.

        :return: The validation summaries.
        """
        return self._logger.validation_summaries

    def get_epochs(self) -> int:
        """
        Get the overall epochs this callback participated in.

        :return: The overall epochs this callback participated in.
        """
        return self._logger.epochs

    def get_train_iterations(self) -> int:
        """
        Get the overall train iterations this callback participated in.

        :return: The overall train iterations this callback participated in.
        """
        return self._logger.training_iterations

    def get_validation_iterations(self) -> int:
        """
        Get the overall validation iterations this callback participated in.

        :return: The overall validation iterations this callback participated in.
        """
        return self._logger.validation_iterations

    def on_horovod_check(self, rank: int) -> bool:
        """
        Check whether this callback is fitting to run by the given horovod rank (worker).

        :param rank: The horovod rank (worker) id.

        :return: True if the callback is ok to run on this rank and false if not.
        """
        return rank == 0

    def on_run_begin(self):
        """
        After the trainer / evaluator run begins, this method will be called to setup the results and hyperparameters
        dictionaries for logging, noting the metrics names and logging the initial hyperparameters values (epoch 0).
        """
        # Setup the results and summaries dictionaries:
        # # Loss:
        self._logger.log_metric(
            metric_name=self._get_metric_name(
                metric_type=self._MetricType.LOSS,
                metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
            )
        )
        # # Metrics:
        for metric_function in self._objects[self._ObjectKeys.METRIC_FUNCTIONS]:
            metric_name = self._get_metric_name(
                metric_type=self._MetricType.ACCURACY, metric_function=metric_function
            )
            self._logger.log_metric(metric_name=metric_name)

        # Setup the hyperparameters dictionaries:
        # # Static hyperparameters:
        if self._static_hyperparameters_keys:
            for name, value in self._static_hyperparameters_keys.items():
                if isinstance(value, Tuple):
                    # Its a parameter that needed to be extracted via key chain.
                    self._logger.log_static_hyperparameter(
                        parameter_name=name,
                        value=self._get_hyperparameter(
                            source=self._objects[value[0]], key_chain=value[1]
                        ),
                    )
                else:
                    # Its a custom hyperparameter.
                    self._logger.log_static_hyperparameter(
                        parameter_name=name, value=value
                    )
        # # Dynamic hyperparameters:
        if self._dynamic_hyperparameters_keys:
            for name, (source, key_chain) in self._dynamic_hyperparameters_keys.items():
                self._logger.log_dynamic_hyperparameter(
                    parameter_name=name,
                    value=self._get_hyperparameter(
                        source=self._objects[source], key_chain=key_chain
                    ),
                )

    def on_epoch_begin(self, epoch: int):
        """
        After the trainer given epoch begins, this method will be called to append a new list to each of the metrics
        results for the new epoch.

        :param epoch: The epoch that is about to begin.
        """
        self._logger.log_epoch()

    def on_epoch_end(self, epoch: int):
        """
        Before the trainer given epoch ends, this method will be called to log the dynamic hyperparameters as needed.

        :param epoch: The epoch that has just ended.
        """
        # Update the dynamic hyperparameters dictionary:
        if self._dynamic_hyperparameters_keys:
            for parameter_name, (
                source_name,
                key_chain,
            ) in self._dynamic_hyperparameters_keys.items():
                self._logger.log_dynamic_hyperparameter(
                    parameter_name=parameter_name,
                    value=self._get_hyperparameter(
                        source=self._objects[source_name], key_chain=key_chain
                    ),
                )

    def on_train_begin(self):
        """
        After the trainer training of the current epoch begins, this method will be called.
        """
        self._log_iteration = False

    def on_validation_begin(self):
        """
        After the trainer / evaluator validation (in a trainer's case it will be per epoch) begins, this method will be
        called.
        """
        self._log_iteration = False

    def on_validation_end(
        self, loss_value: MetricValueType, metric_values: List[float]
    ):
        """
        Before the trainer / evaluator validation (in a trainer's case it will be per epoch) ends, this method will be
        called to log the validation results summaries.

        :param loss_value:    The loss summary of this validation.
        :param metric_values: The metrics summaries of this validation.
        """
        # Store the validation loss average of this epoch:
        self._logger.log_validation_summary(
            metric_name=self._get_metric_name(
                metric_type=self._MetricType.LOSS,
                metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
            ),
            result=float(loss_value),
        )

        # Store the validation metrics averages of this epoch:
        for metric_function, metric_value in zip(
            self._objects[self._ObjectKeys.METRIC_FUNCTIONS], metric_values
        ):
            self._logger.log_validation_summary(
                metric_name=self._get_metric_name(
                    metric_type=self._MetricType.ACCURACY,
                    metric_function=metric_function,
                ),
                result=float(metric_value),
            )

    def on_train_batch_begin(self, batch: int, x: Tensor, y_true: Tensor):
        """
        After the trainer training of the given batch begins, this method will be called to check whether this iteration
        needs to be logged.

        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        """
        self._logger.log_training_iteration()
        self._on_batch_begin(batch=batch)

    def on_validation_batch_begin(self, batch: int, x: Tensor, y_true: Tensor):
        """
        After the trainer / evaluator validation of the given batch begins, this method will be called to check whether
        this iteration needs to be logged.

        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        """
        self._logger.log_validation_iteration()
        self._on_batch_begin(batch=batch)

    def on_train_loss_end(self, loss_value: MetricValueType):
        """
        After the trainer training calculation of the loss, this method will be called to log the loss value.

        :param loss_value: The recent loss value calculated during training.
        """
        # Check if this iteration should be logged:
        if not self._log_iteration:
            return

        # Store the loss value at the current epoch:
        self._logger.log_training_result(
            metric_name=self._get_metric_name(
                metric_type=self._MetricType.LOSS,
                metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
            ),
            result=float(loss_value),
        )

    def on_validation_loss_end(self, loss_value: MetricValueType):
        """
        After the trainer / evaluator validating calculation of the loss, this method will be called to log the loss
        value.

        :param loss_value: The recent loss value calculated during validation.
        """
        # Check if this iteration should be logged:
        if not self._log_iteration:
            return

        # Store the loss value at the current epoch:
        self._logger.log_validation_result(
            metric_name=self._get_metric_name(
                metric_type=self._MetricType.LOSS,
                metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
            ),
            result=float(loss_value),
        )

    def on_train_metrics_end(self, metric_values: List[MetricValueType]):
        """
        After the trainer training calculation of the metrics, this method will be called to log the metrics values.

        :param metric_values: The recent metric values calculated during training.
        """
        # Check if this iteration should be logged:
        if not self._log_iteration:
            return

        # Log the given metrics as needed:
        for metric_function, metric_value in zip(
            self._objects[self._ObjectKeys.METRIC_FUNCTIONS], metric_values
        ):
            self._logger.log_training_result(
                metric_name=self._get_metric_name(
                    metric_type=self._MetricType.ACCURACY,
                    metric_function=metric_function,
                ),
                result=float(metric_value),
            )

    def on_validation_metrics_end(self, metric_values: List[MetricValueType]):
        """
        After the trainer / evaluator validating calculation of the metrics, this method will be called to log the
        metrics values.

        :param metric_values: The recent metric values calculated during validation.
        """
        # Check if this iteration should be logged:
        if not self._log_iteration:
            return

        # Log the given metrics as needed:
        for metric_function, metric_value in zip(
            self._objects[self._ObjectKeys.METRIC_FUNCTIONS], metric_values
        ):
            self._logger.log_validation_result(
                metric_name=self._get_metric_name(
                    metric_type=self._MetricType.ACCURACY,
                    metric_function=metric_function,
                ),
                result=float(metric_value),
            )

    def _on_batch_begin(self, batch: int):
        """
        Method to run on every batch (training and validation).

        :param batch: The batch index.
        """
        self._log_iteration = batch % self._per_iteration_logging == 0

    @staticmethod
    def _get_metric_name(metric_type: str, metric_function: MetricFunctionType):
        """
        Get the given metric name.

        :param metric_type:     Each metric can be either 'loss' or 'accuracy'.
        :param metric_function: The metric function to get its name.

        :return: The metric name.
        """
        if isinstance(metric_function, Module):
            function_name = metric_function.__class__.__name__
        else:
            function_name = metric_function.__name__
        return "{}:{}".format(function_name, metric_type)

    @staticmethod
    def _get_hyperparameter(source, key_chain: List[Union[str, int]]) -> TrackableType:
        """
        Access the hyperparameter from the source using the given key chain.

        :param source:    The object to get the hyperparamter value from.
        :param key_chain: The keys and indices to get to the hyperparameter from the given source object.

        :return: The hyperparameter value.

        :raise KeyError:   In case the one of the keys in the key chain is incorrect.
        :raise IndexError: In case the one of the keys in the key chain is incorrect.
        :raise ValueError: In case the value is not trackable.
        """
        # Get the value using the provided key chain:
        value = source.__dict__
        for key in key_chain:
            try:
                if isinstance(key, int):
                    value = value[key]
                else:
                    value = getattr(value, key)
            except KeyError or IndexError as KeyChainError:
                raise KeyChainError(
                    "Error during getting a hyperparameter value from the {} object. "
                    "The {} in it does not have the following key/index from the key provided: {}"
                    "".format(source.__class__, value.__class__, key)
                )

        # Parse the value:
        if isinstance(value, Tensor) or isinstance(value, Parameter):
            if value.numel() == 1:
                value = float(value)
            else:
                raise ValueError(
                    "The parameter with the following key chain: {} is a pytorch.Tensor with {} elements."
                    "PyTorch tensors are trackable only if they have 1 element."
                    "".format(key_chain, value.numel())
                )
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                value = float(value)
            else:
                raise ValueError(
                    "The parameter with the following key chain: {} is a numpy.ndarray with {} elements."
                    "numpy arrays are trackable only if they have 1 element."
                    "".format(key_chain, value.size)
                )
        elif not (
            isinstance(value, float)
            or isinstance(value, int)
            or isinstance(value, str)
            or isinstance(value, bool)
        ):
            raise ValueError(
                "The parameter with the following key chain: {} is of type '{}'."
                "The only trackable types are: float, int, str and bool."
                "".format(key_chain, type(value))
            )
        return value
