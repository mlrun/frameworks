from typing import Union, List, Dict, Tuple
import numpy as np
from torch import Tensor
from torch.nn import Module
from frameworks.pytorch.callbacks.callback import (
    Callback,
    MetricValueType,
    MetricFunctionType,
)

# All trackable values types:
TrackableType = Union[str, bool, float, int]


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
    Callback for collecting data during training / validation. All the collected data will be available in this callback
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

        # Setup the results dictionary - a dictionary of metrics for all the iteration results by their epochs:
        # [Metric: str] -> [Epoch: int] -> [Iteration: int] -> [value: float]
        self._training_results = {}  # type: Dict[str, List[List[float]]]
        self._validation_results = {}  # type: Dict[str, List[List[float]]]

        # Setup the metrics summary dictionary - a dictionary of all validation metrics averages by epochs:
        # [Metric: str] -> [Epoch: int] -> [value: float]:
        self._summaries = {}  # type: Dict[str, List[float]]

        # Store the static hyperparameters given - a dictionary of parameters and their values to note:
        # [Parameter: str] -> [value: Union[str, bool, float, int]]
        self._static_hyperparameters = {}  # type: Dict[str, TrackableType]

        # Setup the dynamic hyperparameters dictionary - a dictionary of all tracked hyperparameters by epochs:
        # [Hyperparameter: str] -> [Epoch: int] -> [value: Union[str, bool, float, int]]
        self._dynamic_hyperparameters = {}  # type: Dict[str, List[TrackableType]]

        # Setup the iterations counter:
        self._epochs = 0
        self._train_iterations = 0
        self._validation_iterations = 0
        self._log_iteration = False

    @property
    def training_results(self) -> Dict[str, List[List[float]]]:
        """
        Get the training results logged. The results will be stored in a dictionary where each key is the metric name
        and the value is a list of lists of values. The first list is by epoch and the second list is by iteration
        (batch).
        :return: The training results.
        """
        return self._training_results

    @property
    def validation_results(self) -> Dict[str, List[List[float]]]:
        """
        Get the validation results logged. The results will be stored in a dictionary where each key is the metric name
        and the value is a list of lists of values. The first list is by epoch and the second list is by iteration
        (batch).
        :return: The validation results.
        """
        return self._validation_results

    @property
    def static_hyperparameters(self) -> Dict[str, TrackableType]:
        """
        Get the static hyperparameters logged. The hyperparameters will be stored in a dictionary where each key is the
        hyperparameter name and the value is his logged value.
        :return: The static hyperparameters.
        """
        return self._static_hyperparameters

    @property
    def dynamic_hyperparameters(self) -> Dict[str, List[TrackableType]]:
        """
        Get the dynamic hyperparameters logged. The hyperparameters will be stored in a dictionary where each key is the
        hyperparameter name and the value is a list of his logged values per epoch.
        :return: The dynamic hyperparameters.
        """
        return self._dynamic_hyperparameters

    @property
    def summaries(self) -> Dict[str, List[float]]:
        """
        Get the validation summaries of the metrics results. The summaries will be stored in a dictionary where each key
        is the metric names and the value is a list of all the summary values per epoch.
        :return: The validation summaries.
        """
        return self._summaries

    @property
    def epochs(self) -> int:
        """
        Get the overall epochs this callback participated in.
        :return: The overall epochs this callback participated in.
        """
        return self._epochs

    @property
    def train_iterations(self) -> int:
        """
        Get the overall train iterations this callback participated in.
        :return: The overall train iterations this callback participated in.
        """
        return self._train_iterations

    @property
    def validation_iterations(self) -> int:
        """
        Get the overall validation iterations this callback participated in.
        :return: The overall validation iterations this callback participated in.
        """
        return self._validation_iterations

    def on_run_begin(self):
        """
        After the trainer / evaluator run begins, this method will be called to setup the results and hyperparameters
        dictionaries for logging.
        """
        # Setup the results and summaries dictionaries:
        # Loss:
        loss_name = self._get_metric_name(
            metric_type=self._MetricType.LOSS,
            metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
        )
        self._training_results[loss_name] = []
        self._validation_results[loss_name] = []
        self._summaries[loss_name] = []
        # Metrics:
        for metric_function in self._objects[self._ObjectKeys.METRIC_FUNCTIONS]:
            metric_name = self._get_metric_name(
                metric_type=self._MetricType.ACCURACY, metric_function=metric_function
            )
            self._training_results[metric_name] = []
            self._validation_results[metric_name] = []
            self._summaries[metric_name] = []

        # Setup the hyperparameters dictionaries:
        # Static hyperparameters:
        if self._static_hyperparameters_keys:
            for name, value in self._static_hyperparameters_keys.items():
                if isinstance(value, Tuple):
                    # Its a parameter that needed to be extracted via key chain.
                    self._static_hyperparameters[name] = self._get_hyperparameter(
                        source=self._objects[value[0]], key_chain=value[1]
                    )
                else:
                    # Its a custom hyperparameter.
                    self._static_hyperparameters[name] = value
        # Dynamic hyperparameters:
        if self._dynamic_hyperparameters_keys:
            for name, (source, key_chain) in self._dynamic_hyperparameters_keys.items():
                self._dynamic_hyperparameters[name] = [
                    self._get_hyperparameter(
                        source=self._objects[source], key_chain=key_chain
                    )
                ]

    def on_epoch_begin(self, epoch: int):
        """
        After the trainer given epoch begins, this method will be called to append a new list to each of the  metrics
        results for the new epoch.
        :param epoch: The epoch that is about to begin.
        """
        # Add a new epoch to each of the metrics in the results dictionary:
        for results_dictionary in [self._training_results, self._validation_results]:
            for metric in results_dictionary:
                results_dictionary[metric].append([])

    def on_epoch_end(self, epoch: int):
        """
        Before the trainer given epoch ends, this method will be called to log the dynamic hyperparameters as needed.
        :param epoch: The epoch that has just ended.
        """
        self._epochs += 1

        # Update the dynamic hyperparameters dictionary:
        if self._dynamic_hyperparameters_keys:
            for parameter_name, (
                source_name,
                key_chain,
            ) in self._dynamic_hyperparameters_keys.items():
                self._dynamic_hyperparameters[parameter_name].append(
                    self._get_hyperparameter(
                        source=self._objects[source_name], key_chain=key_chain
                    )
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
        loss_name = self._get_metric_name(
            metric_type=self._MetricType.LOSS,
            metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
        )
        self._summaries[loss_name].append(float(loss_value))

        # Store the validation metrics averages of this epoch:
        for metric_function, metric_value in zip(
            self._objects[self._ObjectKeys.METRIC_FUNCTIONS], metric_values
        ):
            self._summaries[
                self._get_metric_name(
                    metric_type=self._MetricType.ACCURACY,
                    metric_function=metric_function,
                )
            ].append(float(metric_value))

    def on_train_batch_begin(self, batch: int, x: Tensor, y_true: Tensor):
        """
        After the trainer training of the given batch begins, this method will be called to check whether this iteration
        needs to be logged.
        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        """
        self._on_batch_begin(batch=batch)

    def on_train_batch_end(self, batch: int, x: Tensor, y_true: Tensor, y_pred: Tensor):
        """
        Before the trainer training of the given batch ends, this method will be called to count the iteration.
        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        :param y_pred: The prediction (output) of the model for this batch's input ('x').
        """
        self._train_iterations += 1

    def on_validation_batch_begin(self, batch: int, x: Tensor, y_true: Tensor):
        """
        After the trainer / evaluator validation of the given batch begins, this method will be called to check whether
        this iteration needs to be logged.
        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        """
        self._on_batch_begin(batch=batch)

    def on_validation_batch_end(
        self, batch: int, x: Tensor, y_true: Tensor, y_pred: Tensor
    ):
        """
        Before the trainer / evaluator validation of the given batch ends, this method will be called to count the
        iteration.
        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        :param y_pred: The prediction (output) of the model for this batch's input ('x').
        """
        self._validation_iterations += 1

    def on_train_loss_end(self, loss_value: MetricValueType):
        """
        After the trainer training calculation of the loss, this method will be called to log the loss value.
        :param loss_value: The recent loss value calculated during training.
        """
        self._on_loss_end(
            results_dictionary=self._training_results, loss_value=loss_value
        )

    def on_validation_loss_end(self, loss_value: MetricValueType):
        """
        After the trainer / evaluator validating calculation of the loss, this method will be called to log the loss
        value.
        :param loss_value: The recent loss value calculated during validation.
        """
        self._on_loss_end(
            results_dictionary=self._validation_results, loss_value=loss_value
        )

    def on_train_metrics_end(self, metric_values: List[MetricValueType]):
        """
        After the trainer training calculation of the metrics, this method will be called to log the metrics values.
        :param metric_values: The recent metric values calculated during training.
        """
        self._on_metrics_end(
            results_dictionary=self._training_results, metric_values=metric_values
        )

    def on_validation_metrics_end(self, metric_values: List[MetricValueType]):
        """
        After the trainer / evaluator validating calculation of the metrics, this method will be called to log the
        metrics values.
        :param metric_values: The recent metric values calculated during validation.
        """
        self._on_metrics_end(
            results_dictionary=self._validation_results, metric_values=metric_values
        )

    def _on_batch_begin(self, batch: int):
        self._log_iteration = batch % self._per_iteration_logging == 0

    def _on_loss_end(self, results_dictionary: dict, loss_value: MetricValueType):
        # Check if this iteration should be logged:
        if not self._log_iteration:
            return

        # Store the loss value at the current epoch:
        loss_name = self._get_metric_name(
            metric_type=self._MetricType.LOSS,
            metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
        )
        results_dictionary[loss_name][-1].append(float(loss_value))

    def _on_metrics_end(
        self, results_dictionary: dict, metric_values: List[MetricValueType]
    ):
        """
        Log the given metrics values to the given results dictionary.
        :param results_dictionary: One of 'self._training_results' or 'self._validation_results'.
        :param metric_values:      The metrics values to log.
        """
        # Check if this iteration should be logged:
        if not self._log_iteration:
            return

        # Log the given metrics as needed:
        for metric_function, metric_value in zip(
            self._objects[self._ObjectKeys.METRIC_FUNCTIONS], metric_values
        ):
            # Get the metric name:
            metric_name = self._get_metric_name(
                metric_type=self._MetricType.ACCURACY, metric_function=metric_function
            )
            # Store the metric value at the current epoch:
            results_dictionary[metric_name][-1].append(float(metric_value))

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
        :raise ValueError: In case the value is not trackable.
        """
        # Get the value using the provided key chain:
        value = source.__dict__
        for key in key_chain:
            try:
                value = value[key]
            except KeyError or IndexError:
                raise KeyError(
                    "Error during getting a hyperparameter value from the {} object. "
                    "The {} in it does not have the following key/index from the keys provided: {}"
                    "".format(source.__class__, value.__class__, key)
                )

        # Parse the value:
        if isinstance(value, Tensor):
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
