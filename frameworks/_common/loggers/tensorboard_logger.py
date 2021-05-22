from typing import Union, Dict, List, Tuple, Callable, TypeVar, Generic
from abc import abstractmethod
import os
from datetime import datetime
import numpy as np
import mlrun
from mlrun import MLClientCtx
from frameworks._common.loggers.logger import Logger
from tensorflow import Variable as TensorflowWeight
from torch.nn import Parameter as PyTorchWeight

# Define a type variable for the different weight holder objects of the supported frameworks:
Weight = TypeVar("Weight", TensorflowWeight, PyTorchWeight)


class TensorboardLogger(Logger, Generic[Weight]):
    """
    An abstract tensorboard logger class for logging the information collected during training / evaluation of the base
    logger to tensorboard. Each framework has its own way of logging to tensorboard, but each must implement the entire
    features listed in this class. The logging includes:

    * Summary text of the run with a hyperlink to the MLRun log if it was done.
    * Hyperparameters tuning table: static hyperparameters, dynamic hyperparameters and epoch validation summaries.
    * Plots:

      * Per iteration (batch) plot for the training and validation metrics.
      * Per epoch plot for the dynamic hyperparameters and validation summaries results.
      * Per epoch weights statistics for each weight and statistic.

    * Histograms per epoch for each of the logged weights.
    * Distributions per epoch for each of the logged weights.
    * Images per epoch for each of the logged weights.
    * Model architecture graph.
    """

    # The default tensorboard directory to be used with a given context:
    _DEFAULT_TENSORBOARD_DIRECTORY = os.path.join(
        os.sep, "User", ".tensorboard", "{{project}}"
    )

    class _Sections:
        """
        Tensorboard split his plots to sections via a path like name <SECTION>/<PLOT_NAME>. These are the sections used
        in this callback for logging.
        """

        TRAINING = "Training"
        VALIDATION = "Validation"
        SUMMARY = "Summary"
        HYPERPARAMETERS = "Hyperparameters"
        WEIGHTS = "Weights"

    def __init__(
        self,
        statistics_functions: List[
            Callable[[Union[Weight]], Union[float, Weight]]
        ],
        context: MLClientCtx = None,
        tensorboard_directory: str = None,
        run_name: str = None,
    ):
        """
        Initialize a tensorboard logger callback with the given configuration. At least one of 'context' and
        'tensorboard_directory' must be given.

        :param statistics_functions:  A list of statistics functions to calculate at the end of each epoch on the
                                      tracked weights. Only relevant if weights are being tracked. The functions in
                                      the list must accept one Weight and return a float (or float convertible) value.
        :param context:               A mlrun context to use for logging into the user's tensorboard directory.
        :param tensorboard_directory: If context is not given, or if wished to set the directory even with context,
                                      this will be the output for the event logs of tensorboard.
        :param run_name:              This experiment run name. Each run name will be indexed at the end of the name so
                                      each experiment will be numbered automatically. If a context was given, the
                                      context's uid will be added instead of an index. If a run name was not given the
                                      current time in the following format: 'YYYY-mm-dd_HH:MM:SS'.
        """
        super(TensorboardLogger, self).__init__()

        # Store the context and statistics functions:
        self._context = context
        self._statistics_functions = statistics_functions

        # Create the output path to work with:
        self._output_path, self._run_name = self._create_output_path(
            tensorboard_directory=tensorboard_directory, run_name=run_name
        )

        # Setup the weights dictionaries - a dictionary of all required weight parameters:
        # [Weight: str] -> [value: WeightType]
        self._weights = {}  # type: Dict[str, Weight]

        # Setup the statistics dictionaries - a dictionary of statistics for the required weights per epoch:
        # [Statistic: str] -> [Weight: str] -> [epoch: int] -> [value: float]
        self._weights_statistics = {}  # type: Dict[str, Dict[str, List[float]]]
        for statistic_function in self._statistics_functions:
            self._weights_statistics[statistic_function.__name__] = {}  # type: Dict[str, List[float]]

    @property
    def weights(self):
        """
        Get the logged weights dictionary. Each of the logged weight will be found by its name.

        :return: The weights dictionary.
        """
        return self._weights

    @property
    def weight_statistics(self):
        """
        Get the logged statistics for all the tracked weights. Each statistic has a dictionary of weights and their list
        of epochs values.

        :return: The statistics dictionary.
        """
        return self._weights_statistics

    def log_weight(self, weight_name: str, weight_holder: Weight):
        """
        Log the weight into the weights dictionary so it will be tracked and logged during the epochs. For each logged
        weight the key for it in the statistics logged will be initialized as well.

        :param weight_name:   The weight's name.
        :param weight_holder: The weight holder to track. Both Tensorflow (including Keras) and PyTorch (including
                              Lightning) keep the weights tensor in a holder object - 'Variable' for Tensorflow and
                              'Parameter' for PyTorch.
        """
        # Collect the given weight:
        self._weights[weight_name] = weight_holder

        # Insert the weight to all the statistics:
        for statistic in self._weights_statistics:
            self._weights_statistics[statistic][weight_name] = []

    def log_weights_statistics(self):
        """
        Calculate the statistics on the current weights and log the results.
        """
        for weight_name, weight_parameter in self._weights.items():
            for statistic_function in self._statistics_functions:
                self._weights_statistics[statistic_function.__name__][
                    weight_name
                ].append(float(statistic_function(weight_parameter)))

    @abstractmethod
    def log_context_summary_to_tensorboard(self):
        """
        Log a summary of this training / validation run to tensorboard.
        """
        pass

    @abstractmethod
    def log_parameters_table_to_tensorboard(self):
        """
        Log the validation summaries, static and dynamic hyperparameters to the 'HParams' table in tensorboard.
        """
        pass

    @abstractmethod
    def log_training_results_to_tensorboard(self):
        """
        Log the recent training iteration metrics results to tensorboard.
        """
        pass

    @abstractmethod
    def log_validation_results_to_tensorboard(self):
        """
        Log the recent validation iteration metrics results to tensorboard.
        """
        pass

    @abstractmethod
    def log_dynamic_hyperparameters_to_tensorboard(self):
        """
        Log the recent epoch dynamic hyperparameters values to tensorboard.
        """
        pass

    @abstractmethod
    def log_summaries_to_tensorboard(self):
        """
        Log the recent epoch summaries results to tensorboard.
        """
        pass

    @abstractmethod
    def log_weights_histograms_to_tensorboard(self):
        """
        Log the current state of the weights as histograms to tensorboard.
        """
        pass

    @abstractmethod
    def log_weights_images_to_tensorboard(self):
        """
        Log the current state of the weights as images to tensorboard.
        """
        pass

    @abstractmethod
    def log_statistics_to_tensorboard(self):
        """
        Log the last stored statistics values this logger collected to tensorboard.
        """
        pass

    @abstractmethod
    def log_model_to_tensorboard(self, *args, **kwargs):
        """
        Log the given model as a graph in tensorboard.
        """
        pass

    def _create_output_path(
        self, tensorboard_directory: str, run_name: str
    ) -> Tuple[str, str]:
        """
        Create the output path, indexing the given run name as needed.

        :param tensorboard_directory: The output directory for the event files to be logged to. In case a context was
                                      given this value can be 'None' so it will save to the user's default directory.
        :param run_name:              The run name to be indexed if needed. If a context was given, the run name will
                                      include the context's uid.

        :return: A tuple of strings:
                 [0] = The created output path.
                 [1] = The edited run name.
        """
        # If a run name was not given, take the current timestamp as the run name in the format 'YYYY-mm-dd_HH:MM:SS':
        if run_name is None:
            run_name = (
                str(datetime.now()).split(".")[0].replace(" ", "_")
                if (self._context is None or self._context.name != "")
                else self._context.name
            )

        # Check if a context is available:
        if self._context:
            # Try to get the 'tensorboard_dir' parameter:
            tensorboard_directory = self._context.get_param("tensorboard_dir")
            if tensorboard_directory is None:
                # The parameter was not given, set the directory to the default value:
                tensorboard_directory = self._DEFAULT_TENSORBOARD_DIRECTORY.replace(
                    "{{project}}", self._context.project
                )
        else:
            # Create the main tensorboard directory:
            os.makedirs(tensorboard_directory, exist_ok=True)
            # Index the run name according to the tensorboard directory content:
            index = 1
            for run_directory in sorted(os.listdir(tensorboard_directory)):
                existing_run = run_directory.rsplit(
                    "_", 1
                )  # type: List[str] # [0] = name, [1] = index
                if run_name == existing_run[0]:
                    index = int(existing_run[1]) + 1
            # Build the full run name:
            run_name = "{}_{}".format(run_name, index)

        # Create the output path:
        output_path = os.path.join(tensorboard_directory, run_name)
        os.makedirs(output_path)

        return output_path, run_name

    def _parse_context_summary(self) -> str:
        """
        Parse and return the run summary string to log into tensorboard as text.

        :return: The run summary string.
        """
        # TODO: take a look in mlrun/mlrun/render.py and see the uid_template html hyperlink and put it the text
        #       summary. Add context meta-data.
        return "Testing testing"
