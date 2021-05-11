from typing import Dict, Tuple, List, Union, Any
import numpy as np
import tensorflow as tf
from tensorflow import Tensor, Variable
from tensorflow import keras
from tensorflow.keras.callbacks import Callback

# All trackable values types:
TrackableType = Union[str, bool, float, int]


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
        dynamic_hyperparameters: Dict[str, List[Union[str, int]]] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, List[Union[str, int]]]
        ] = None,
        per_iteration_logging: int = 1,
    ):
        """
        Initialize a logging callback with the given hyperparameters and logging configurations.
        :param dynamic_hyperparameters: If needed to track a hyperparameter dynamically (sample it each epoch) it should
                                        be passed here. The parameter expects a dictionary where the keys are the
                                        hyperparameter chosen names and the values are a key chain. A key chain is a
                                        list of keys and indices to know how to access the needed hyperparameter. For
                                        example, to track the 'lr' attribute of an optimizer, one should pass:
                                        {
                                            "learning rate": ["optimizer", "lr"]
                                        }
        :param static_hyperparameters:  If needed to track a hyperparameter one time per run it should be passed here.
                                        The parameter expects a dictionary where the keys are the
                                        hyperparameter chosen names and the values are the hyperparameter static value
                                        or a key chain just like the dynamic hyperparameter. For example, to track the
                                        'epochs' of an experiment run, one should pass:
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
        # [Metric: str] -> [Epoch: int] -> [Iteration: int] -> [Value: float]
        self._training_results = {}  # type: Dict[str, List[List[float]]]
        self._validation_results = {}  # type: Dict[str, List[List[float]]]

        # For calculating the batch's values we need to collect the epochs sums:
        # [Metric: str] -> [Sum: float]
        self._training_epoch_sums = {}  # type: Dict[str, float]
        self._validation_epoch_sums = {}  # type: Dict[str, float]

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

        # Set the setup_tun flag:
        self._run_set_up = False

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

    def on_train_begin(self, logs: dict = None):
        """
        Called once at the beginning of training process (one time call).
        :param logs: Dict. Currently no data is passed to this argument for this method but that may change in the
                     future.
        """
        self._setup_run()

    def on_test_begin(self, logs: dict = None):
        """
        Called at the beginning of evaluation or validation. Will be called on each epoch according to the validation
        per epoch configuration.
        :param logs: Dict. Currently no data is passed to this argument for this method but that may change in the
                     future.
        """
        # If this callback is part of evaluation and not training, need to check if the run was setup:
        if not self._run_set_up:
            self._setup_run()

    def on_test_end(self, logs: Dict[str, Any] = None):
        """
        Called at the end of evaluation or validation. Will be called on each epoch according to the validation
        per epoch configuration.
        :param logs: Dict. Currently no data is passed to this argument for this method but that may change in the
        future.
        """
        # Store the metrics average of this epoch:
        for metric_name, epoch_values in self._validation_results.items():
            # Check if needed to initialize:
            if metric_name not in self._summaries:
                self._summaries[metric_name] = []
            self._summaries[metric_name].append(
                float(sum(epoch_values[-1]) / len(epoch_values[-1]))
            )

    def on_epoch_begin(self, epoch: int, logs=None):
        """
        Called at the start of an epoch.
        :param epoch: Integer, index of epoch.
        :param logs:  Dict. Currently no data is passed to this argument for this method but that may change in the
                      future.
        """
        # Add a new epoch to each of the metrics in the results dictionary and reset the metric sum:
        for results_dictionary, sum_dictionary in zip(
            [self._training_results, self._validation_results],
            [self._training_epoch_sums, self._validation_epoch_sums],
        ):
            for metric in results_dictionary:
                results_dictionary[metric].append([])
                sum_dictionary[metric] = 0

    def on_epoch_end(self, epoch: int, logs: Dict[str, TrackableType] = None):
        """
        Called at the end of an epoch.
        :param epoch: Integer, index of epoch.
        :param logs:  Dict, metric results for this training epoch, and for the validation epoch if validation is
                      performed. Validation result keys are prefixed with `val_`. For training epoch, the values of the
                      `Model`'s metrics are returned. Example : `{'loss': 0.2, 'acc': 0.7}`.
        """
        self._epochs += 1

        # Update the dynamic hyperparameters dictionary:
        if self._dynamic_hyperparameters_keys:
            for parameter_name, key_chain in self._dynamic_hyperparameters_keys.items():
                self._dynamic_hyperparameters[parameter_name].append(
                    self._get_hyperparameter(key_chain=key_chain)
                )

    def on_train_batch_begin(self, batch: int, logs: Dict[str, TrackableType] = None):
        """
        Called at the beginning of a training batch in `fit` methods. Note that if the `steps_per_execution` argument to
        `compile` in `tf.keras.Model` is set to `N`, this method will only be called every `N` batches.
        :param batch: Integer, index of batch within the current epoch.
        :param logs:  Dict, contains the return value of `model.train_step`. Typically, the values of the `Model`'s
                      metrics are returned.  Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        self._on_batch_begin(batch=batch)

    def on_train_batch_end(self, batch: int, logs: dict = None):
        """
        Called at the end of a training batch in `fit` methods. Note that if the `steps_per_execution` argument to
        `compile` in `tf.keras.Model` is set to `N`, this method will only be called every `N` batches.
        :param batch: Integer, index of batch within the current epoch.
        :param logs:  Dict. Aggregated metric results up until this batch.
        """
        self._on_batch_end(
            results_dictionary=self._training_results,
            sum_dictionary=self._training_epoch_sums,
            logs=logs,
        )
        self._train_iterations += 1

    def on_test_batch_begin(self, batch: int, logs: dict = None):
        """
        Called at the beginning of a batch in `evaluate` methods. Also called at the beginning of a validation batch in
        the `fit` methods, if validation data is provided. Note that if the `steps_per_execution` argument to `compile`
        in `tf.keras.Model` is set to `N`, this method will only be called every `N` batches.

        :param batch: Integer, index of batch within the current epoch.
        :param logs:  Dict, contains the return value of `model.test_step`. Typically, the values of the `Model`'s
                      metrics are returned.  Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        self._on_batch_begin(batch=batch)

    def on_test_batch_end(self, batch: int, logs: dict = None):
        """
        Called at the end of a batch in `evaluate` methods. Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided. Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N` batches.
        :param batch: Integer, index of batch within the current epoch.
        :param logs:  Dict. Aggregated metric results up until this batch.
        """
        self._on_batch_end(
            results_dictionary=self._validation_results,
            sum_dictionary=self._validation_epoch_sums,
            logs=logs,
        )
        self._validation_iterations += 1

    def _setup_run(self):
        """
        After the trainer / evaluator run begins, this method will be called to setup the results and hyperparameters
        dictionaries for logging.
        """
        # Setup the hyperparameters dictionaries:
        # Static hyperparameters:
        if self._static_hyperparameters_keys:
            for name, value in self._static_hyperparameters_keys.items():
                if isinstance(value, List):
                    # Its a parameter that needed to be extracted via key chain.
                    self._static_hyperparameters[name] = self._get_hyperparameter(
                        key_chain=value
                    )
                else:
                    # Its a custom hyperparameter.
                    self._static_hyperparameters[name] = value
        # Dynamic hyperparameters:
        if self._dynamic_hyperparameters_keys:
            for name, key_chain in self._dynamic_hyperparameters_keys.items():
                self._dynamic_hyperparameters[name] = [
                    self._get_hyperparameter(key_chain=key_chain)
                ]

        # Mark this run was set up:
        self._run_set_up = True

    def _on_batch_begin(self, batch: int):
        """
        Method to run on every batch (training and validation).
        :param batch: The batch index.
        """
        self._log_iteration = batch % self._per_iteration_logging == 0

    def _on_batch_end(self, results_dictionary: dict, sum_dictionary: dict, logs: dict):
        """
        Log the given metrics values to the given results dictionary.
        :param results_dictionary: One of 'self._training_results' or 'self._validation_results'.
        :param sum_dictionary:     One of 'self._training_epoch_sums' or 'self._validation_epoch_sums'.
        :param logs:               The loss and metrics results of the recent batch.
        """
        # Check if this iteration should be logged:
        if not self._log_iteration:
            return

        # Log the given metrics as needed:
        for metric_name_in_log, aggregated_value in logs.items():
            # Get the metric name:
            metric_name = self._get_metric_name(metric_name_in_log=metric_name_in_log)
            # Check if needed to initialize:
            if metric_name not in results_dictionary:
                results_dictionary[metric_name] = [[]]
                sum_dictionary[metric_name] = 0
            # Calculate the last value:
            elements_number = len(results_dictionary[metric_name][-1]) + 1
            elements_sum = sum_dictionary[metric_name]
            last_metric_value = elements_number * aggregated_value - elements_sum
            # Store the metric value at the current epoch:
            sum_dictionary[metric_name] += last_metric_value
            results_dictionary[metric_name][-1].append(float(last_metric_value))

    def _get_hyperparameter(self, key_chain: List[Union[str, int]]) -> TrackableType:
        """
        Access the hyperparameter from the model stored in this callback using the given key chain.
        :param key_chain: The keys and indices to get to the hyperparameter from the given source object.
        :return: The hyperparameter value.
        :raise KeyError:   In case the one of the keys in the key chain is incorrect.
        :raise IndexError: In case the one of the keys in the key chain is incorrect.
        :raise ValueError: In case the value is not trackable.
        """
        # Get the value using the provided key chain:
        value = self.model
        for key in key_chain:
            try:
                if isinstance(key, int):
                    value = value[key]
                else:
                    value = getattr(value, key)
            except KeyError or IndexError as KeyChainError:
                raise KeyChainError(
                    "Error during getting a hyperparameter value with the key chain {}. "
                    "The {} in it does not have the following key/index from the key provided: {}"
                    "".format(key_chain, value.__class__, key)
                )

        # Parse the value:
        if isinstance(value, Tensor) or isinstance(value, Variable):
            if int(tf.size(value)) == 1:
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

    def _get_metric_name(self, metric_name_in_log: str):
        """
        Get the given metric name.
        :param metric_name_in_log: The metric function name given from the 'logs' dictionary.
        :return: The metric name.
        """
        metric_type = self._MetricType.ACCURACY
        if metric_name_in_log.startswith("val_"):
            metric_name_in_log = metric_name_in_log.split("val_")[1]
        if metric_name_in_log == "loss":
            metric_name_in_log = self.model.loss
            metric_type = self._MetricType.LOSS
        return "{}:{}".format(metric_name_in_log, metric_type)
