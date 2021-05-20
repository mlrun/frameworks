from typing import Dict, List, Union


# All trackable values types:
TrackableType = Union[str, bool, float, int]


class Logger:
    """
    Logger for tracking hyperparamters and metrics results during training / evaluation of some framework.
    """

    def __init__(self):
        """
        Initialize a generic logger for collecting trainig / validation runs data.
        """
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
        self._training_iterations = 0
        self._validation_iterations = 0

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
        Get the overall epochs.
        :return: The overall epochs.
        """
        return self._epochs

    @property
    def train_iterations(self) -> int:
        """
        Get the overall train iterations.
        :return: The overall train iterations.
        """
        return self._training_iterations

    @property
    def validation_iterations(self) -> int:
        """
        Get the overall validation iterations.
        :return: The overall validation iterations.
        """
        return self._validation_iterations

    def log_epoch(self):
        """
        Log a new epoch, appending all the result with a new list for the new epoch.
        """
        # Count the new epoch:
        self._epochs += 1

        # Add a new epoch to each of the metrics in the results dictionary:
        for results_dictionary in [self._training_results, self._validation_results]:
            for metric in results_dictionary:
                results_dictionary[metric].append([])

    def log_training_iteration(self):
        """
        Log a new training iteration.
        """
        self._training_iterations += 1

    def log_validation_iteration(self):
        """
        Log a new validation iteration.
        """
        self._validation_iterations += 1

    def log_metric(self, metric_name: str):
        """
        Log a new metric, noting it in the results and summary dictionaries.
        :param metric_name: The metric name to log.
        """
        self._training_results[metric_name] = []
        self._validation_results[metric_name] = []
        self._summaries[metric_name] = []

    def log_training_result(self, metric_name: str, result: float):
        """
        Log the given metric result in the training results dictionary at the current epoch.
        :param metric_name: The metric name as it was logged in 'log_metric'.
        :param result:      The metric result to log.
        """
        self._training_results[metric_name][-1].append(result)

    def log_validation_result(self, metric_name: str, result: float):
        """
        Log the given metric result in the validation results dictionary at the current epoch.
        :param metric_name: The metric name as it was logged in 'log_metric'.
        :param result:      The metric result to log.
        """
        self._validation_results[metric_name][-1].append(result)

    def log_summary(self, metric_name: str, result: float):
        """
        Log the given metric result in the summaries results dictionary.
        :param metric_name: The metric name as it was logged in 'log_metric'.
        :param result:      The metric result to log.
        """
        self._summaries[metric_name].append(result)

    def log_static_hyperparameter(self, parameter_name: str, value: TrackableType):
        """
        Log the given parameter value in the static hyperparameters dictionary.
        :param parameter_name: The parameter name.
        :param value:          The parameter value to log.
        """
        self._static_hyperparameters[parameter_name] = value

    def log_dynamic_hyperparameter(self, parameter_name: str, value: TrackableType):
        """
        Log the given parameter value in the dynamic hyperparameters dictionary at the current epoch (if its a new
        parameter it will be epoch 0).
        :param parameter_name: The parameter name.
        :param value:          The parameter value to log.
        """
        if parameter_name not in self._dynamic_hyperparameters:
            self._dynamic_hyperparameters[parameter_name] = [value]
        else:
            self._dynamic_hyperparameters[parameter_name].append(value)