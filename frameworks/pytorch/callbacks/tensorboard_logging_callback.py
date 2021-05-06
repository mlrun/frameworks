from typing import List, Tuple, Dict, Union
import datetime
import os
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from pytorch.callbacks.logging_callback import (
    LoggingCallback,
    TrackableType,
    MetricFunctionType,
)
import mlrun


class _MLRunSummaryWriter(SummaryWriter):
    """
    A slightly edited torch's SummaryWriter class to overcome the hyperparameter logging problem (creating a new event
    per call).
    """

    def add_hparams(
        self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
    ):
        """
        Log the given hyperparameters to the same event file that is currently open.
        :param hparam_dict:            The static hyperparameters to simply log to the 'hparams' table.
        :param metric_dict:            The metrics and dynamic hyper parameters to link with the plots.
        :param hparam_domain_discrete: Not used in this SummaryWriter.
        :param run_name:               Not used in this SummaryWriter.
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)
        self._get_file_writer().add_summary(exp)
        self._get_file_writer().add_summary(ssi)
        self._get_file_writer().add_summary(sei)


class TensorboardLoggingCallback(LoggingCallback):
    """
    Callback for logging data during training / validation to tensorboard. Each tracked metrics results will be logged
    per iteration (batch) and each tracked dynamic hyperparameter and summaries will be logged per epoch. At the end of
    the run the model will be logged as well. In addition, weights histograms, distributions and statistics will be
    logged as well per epoch.

    To summerize, the available data in tensorboard will be:
        * Plots:
            - Training loss and metrics results per iteration (when used with a trainer).
            - Validation loss and metrics results per iteration.
            - Epoch's summaries for loss, metrics and tracked dynamic hyperparameters.
            - Tracked weights statistics per epoch - STD and Mean.
        * Tracked weights:
            - Histograms per epoch.
            - Distributions per epoch.
        * Hyperparameters table:
            - Static hyperparameters.
            - Dynamic hyperparameters per epoch linked to their plot.
        * Model's graph.

    All the collected data will be available in this callback post the training / validation process and can be accessed
    via the 'training_results', 'validation_results', 'static_hyperparameters', 'dynamic_hyperparameters', 'summaries',
    'weights', 'weights_mean' and 'weights_std' properties.
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

    class _StatisticsTypes:
        """
        All of the statistics types calculated each epoch on the tracked weights.
        """

        MEAN = "Mean"
        STD = "STD"

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        tensorboard_directory: str = None,
        run_name: str = None,
        weights: Union[bool, List[str]] = False,
        dynamic_hyperparameters: Dict[str, Tuple[str, List[Union[str, int]]]] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, Tuple[str, List[Union[str, int]]]]
        ] = None,
        per_iteration_logging: int = 1,
    ):
        """
        Initialize a tensorboard logging callback with the given weights, hyperparameters and logging configurations.
        Note that at least one of 'context' and 'tensorboard_directory' must be given.
        :param context:                 A mlrun context to use for logging into the user's tensorboard directory.
        :param tensorboard_directory:   If context is not given, or if wished to set the directory even with context,
                                        this will be the output for the event logs of tensorboard.
        :param run_name:                This experiment run name. Each run name will be indexed at the end of the name
                                        so each experiment will be numbered automatically. If a context was given, the
                                        context's uid will be added instead of an index. If a run name was not given the
                                        current time in the following format: 'YYYY-mm-dd_HH:MM:SS'.
        :param weights:                 If wished to track weights to draw their histograms and calculate statistics per
                                        epoch, the weights names should be passed here. Note that each name given will
                                        be searched as 'if <NAME> in <WEIGHT_NAME>' so a simple module name will be
                                        enough to catch his weights. A boolean value can be passed to track all weights.
                                        Defaulted to False.
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
        :raise ValueError: In case both 'context' and 'tensorboard_directory' parameters were not given.
        """
        super(TensorboardLoggingCallback, self).__init__(
            dynamic_hyperparameters=dynamic_hyperparameters,
            static_hyperparameters=static_hyperparameters,
            per_iteration_logging=per_iteration_logging,
        )
        # Validate input:
        if context is None and tensorboard_directory is None:
            raise ValueError(
                "The {} expect to receive a mlrun.MLClientCtx context or a path to a directory to output"
                "the logging file but None were given.".format(self.__class__)
            )

        # If a run name was not given, take the current timestamp as the run name in the format 'YYYY-mm-dd_HH:MM:SS':
        if run_name is None:
            run_name = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")

        # Save the configurations:
        self._tracked_weights = weights

        # Setup the weights dictionaries - a dictionary of all required weight parameters:
        # [Weight: str] -> [value: Parameter]
        self._weights = {}  # type: Dict[str, Parameter]

        # Setup the weights statistics dictionaries - dictionaries of statistics for all the required weights per epoch:
        # [Weight: str] -> [epoch: int] -> [value: float]
        self._weights_mean = {}  # type: Dict[str, List[float]]
        self._weights_std = {}  # type: Dict[str, List[float]]

        # Store the context:
        self._context = context

        # Get the indexed run name and the output path:
        self._run_name, self._output_path = self._create_output_path(
            tensorboard_directory=tensorboard_directory, run_name=run_name
        )

        # Initialize the tensorboard writer:
        self._summary_writer = _MLRunSummaryWriter(log_dir=self._output_path)

    @property
    def weights(self) -> Dict[str, Parameter]:
        """
        Get the weights tensors tracked. The weights will be stored in a dictionary where each key is the weight's name
        and the value is the weight's parameter (tensor).
        :return: The weights.
        """
        return self._weights

    @property
    def weights_mean(self) -> Dict[str, List[float]]:
        """
        Get the weights mean results logged. The results will be stored in a dictionary where each key is the weight's
        name and the value is a list of mean values per epoch.
        :return: The weights mean results.
        """
        return self._weights_mean

    @property
    def weights_std(self) -> Dict[str, List[float]]:
        """
        Get the weights std results logged. The results will be stored in a dictionary where each key is the weight's
        name and the value is a list of std values per epoch.
        :return: The weights std results.
        """
        return self._weights_std

    def on_setup(
        self,
        model: Module = None,
        training_set: DataLoader = None,
        validation_set: DataLoader = None,
        loss_function: Module = None,
        optimizer: Optimizer = None,
        metric_functions: List[MetricFunctionType] = None,
        scheduler=None,
    ):
        """
        Storing all the given objects in the callback's objects dictionary and load the weights parameters as given in
        the callback's initialization.
        :param model:            The model to be stored in this callback.
        :param training_set:     The training set to be stored in this callback.
        :param validation_set:   The validation set to be stored in this callback.
        :param loss_function:    The loss function to be stored in this callback.
        :param optimizer:        The optimizer to be stored in this callback.
        :param metric_functions: The metric functions to be stored in this callback.
        :param scheduler:        The scheduler to be stored in this callback.
        """
        super(TensorboardLoggingCallback, self).on_setup(
            model=model,
            training_set=training_set,
            validation_set=validation_set,
            loss_function=loss_function,
            optimizer=optimizer,
            metric_functions=metric_functions,
            scheduler=scheduler,
        )

        # Collect the weights for drawing histograms according to the stored configuration:
        if self._tracked_weights is False:
            return
        for weight_name, weight_parameter in self._objects[
            self._ObjectKeys.MODEL
        ].named_parameters():
            if self._tracked_weights is True:  # Collect all weights
                self._weights[weight_name] = weight_parameter
                self._weights_mean[weight_name] = [float(torch.mean(weight_parameter))]
                self._weights_std[weight_name] = [float(torch.std(weight_parameter))]
                continue
            for tag in self._tracked_weights:  # Collect by given name
                if tag in weight_name:
                    self._weights[weight_name] = weight_parameter
                    self._weights_mean[weight_name] = [
                        float(torch.mean(weight_parameter))
                    ]
                    self._weights_std[weight_name] = [
                        float(torch.std(weight_parameter))
                    ]
                    break

    def on_run_begin(self):
        """
        After the trainer / evaluator run begins, this method will be called to setup the weights, results and
        hyperparameters dictionaries for logging. Epoch 0 (pre-run state) will be logged here.
        """
        # Setup all the results and hyperparameters dictionaries:
        super(TensorboardLoggingCallback, self).on_run_begin()

        # Check if needed to track hyperparameters:
        if (
            len(self._static_hyperparameters) == 0
            and len(self._dynamic_hyperparameters) == 0
        ):
            return

        # Prepare the hyperparameters values:
        non_graph_parameters = {}
        for parameter, value in self._static_hyperparameters.items():
            non_graph_parameters[parameter] = value

        # Prepare the summaries values and the dynamic hyperparameters values:
        graph_parameters = {}
        for metric in self._summaries:
            graph_parameters["{}/{}".format(self._Sections.SUMMARY, metric)] = 0.0
        for parameter, epochs in self._dynamic_hyperparameters.items():
            graph_parameters[
                "{}/{}".format(self._Sections.HYPERPARAMETERS, parameter)
            ] = epochs[-1]

        # Write the hyperparameters and summaries table:
        self._summary_writer.add_hparams(non_graph_parameters, graph_parameters)

        # Add initial dynamic hyperparameters values (epoch 0) to their graphs:
        for parameter, epochs in self._dynamic_hyperparameters.items():
            self._summary_writer.add_scalar(
                tag="{}/{}".format(self._Sections.HYPERPARAMETERS, parameter),
                scalar_value=epochs[-1],
                global_step=0,
            )

        # Draw the initial weights (epoch 0) graphs:
        for weight_name, weight_parameter in self._weights.items():
            # Draw histogram:
            self._summary_writer.add_histogram(
                tag="{}/{}".format(self._Sections.WEIGHTS, weight_name),
                values=weight_parameter,
                global_step=0,
            )
            # Add Mean value:
            self._summary_writer.add_scalar(
                tag="{}/{}:{}".format(
                    self._Sections.WEIGHTS, weight_name, self._StatisticsTypes.MEAN
                ),
                scalar_value=self._weights_mean[weight_name][-1],
                global_step=0,
            )
            # Add Standard-deviation value:
            self._summary_writer.add_scalar(
                tag="{}/{}:{}".format(
                    self._Sections.WEIGHTS, weight_name, self._StatisticsTypes.STD
                ),
                scalar_value=self._weights_std[weight_name][-1],
                global_step=0,
            )

    def on_run_end(self):
        """
        Before the trainer / evaluator run ends, this method will be called to log the model's graph.
        """
        # Store the trained model:
        self._summary_writer.add_graph(
            model=self._objects[self._ObjectKeys.MODEL],
            input_to_model=next(
                self._objects[self._ObjectKeys.TRAINING_SET].__iter__()
            )[0],
        )

    def on_epoch_end(self, epoch: int):
        """
        Before the trainer given epoch ends, this method will be called to log the dynamic hyperparameters as needed.
        All of the per epoch plots (loss and metrics summaries, dynamic hyperparameters, weights histograms and
        statistics) will log this epoch's tracked values.
        :param epoch: The epoch that has just ended.
        """
        super(TensorboardLoggingCallback, self).on_epoch_end(epoch=epoch)

        # Add this epoch loss and metrics averages to their graphs:
        for parameter, epochs in self._summaries.items():
            self._summary_writer.add_scalar(
                tag="{}/{}".format(self._Sections.SUMMARY, parameter),
                scalar_value=epochs[-1],
                global_step=self._epochs,
            )

        # Add this epoch dynamic hyperparameters values to their graphs:
        for parameter, epochs in self._dynamic_hyperparameters.items():
            self._summary_writer.add_scalar(
                tag="{}/{}".format(self._Sections.HYPERPARAMETERS, parameter),
                scalar_value=epochs[-1],
                global_step=self._epochs,
            )

        # Add weight histograms and statistics for all the tracked weights:
        for weight_name, weight_parameter in self._weights.items():
            # Draw histogram:
            self._summary_writer.add_histogram(
                tag="{}/{}".format(self._Sections.WEIGHTS, weight_name),
                values=weight_parameter,
                global_step=self._epochs,
            )
            # Add Mean value:
            self._weights_mean[weight_name].append(float(torch.mean(weight_parameter)))
            self._summary_writer.add_scalar(
                "{}/{}:{}".format(
                    self._Sections.WEIGHTS, weight_name, self._StatisticsTypes.MEAN
                ),
                scalar_value=self._weights_mean[weight_name][-1],
                global_step=self._epochs,
            )
            # Add STD value:
            self._weights_std[weight_name].append(float(torch.std(weight_parameter)))
            self._summary_writer.add_scalar(
                "{}/{}:{}".format(
                    self._Sections.WEIGHTS, weight_name, self._StatisticsTypes.STD
                ),
                scalar_value=self._weights_std[weight_name][-1],
                global_step=self._epochs,
            )

        # Make sure all values were written to the directory logs:
        self._summary_writer.flush()

    def on_train_batch_end(self, batch: int, x: Tensor, y_true: Tensor, y_pred: Tensor):
        """
        Before the trainer training of the given batch ends, this method will be called to log the batch's loss and
        metrics results to their per iteration plots.
        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        :param y_pred: The prediction (output) of the model for this batch's input ('x').
        """
        # Check if Tensorboard is needed:
        if not self._summary_writer:
            return

        # Add this batch loss and metrics results to their graphs:
        for parameter, epochs in self._training_results.items():
            self._summary_writer.add_scalar(
                tag="{}/{}".format(self._Sections.TRAINING, parameter),
                scalar_value=epochs[-1][-1],
                global_step=self._train_iterations,
            )

        super(TensorboardLoggingCallback, self).on_train_batch_end(
            batch=batch, x=x, y_true=y_true, y_pred=y_pred
        )

    def on_validation_batch_end(
        self, batch: int, x: Tensor, y_true: Tensor, y_pred: Tensor
    ):
        """
        Before the trainer / evaluator validation of the given batch ends, this method will be called to log the batch's
        loss and metrics results to their per iteration plots.
        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        :param y_pred: The prediction (output) of the model for this batch's input ('x').
        """
        # Check if Tensorboard is needed:
        if not self._summary_writer:
            return

        # Add this batch loss and metrics results to their graphs:
        for parameter, epochs in self._validation_results.items():
            self._summary_writer.add_scalar(
                tag="{}/{}".format(self._Sections.VALIDATION, parameter),
                scalar_value=epochs[-1][-1],
                global_step=self._validation_iterations,
            )

        # Count the iteration:
        super(TensorboardLoggingCallback, self).on_validation_batch_end(
            batch=batch, x=x, y_true=y_true, y_pred=y_pred
        )

    def _create_output_path(
        self, tensorboard_directory: str, run_name: str
    ) -> Tuple[str, str]:
        """
        Create the output path, indexing the given run name as needed.
        :param tensorboard_directory: The output directory for the event files to be logged to. In case a context was
                                      given this value can be 'None' so it will save to the user's default directory.
        :param run_name:              The run name to be indexed if needed. If a context was given, the run name will
                                      include the context's uid.
        :return: A tuple of:
                 [0] = Indexed run name.
                 [1] = The full output path.
        """
        # Check if a context is available:
        if self._context:
            # Try to get the 'tensorboard_dir' parameter:
            tensorboard_directory = self._context.get_param("tensorboard_dir")
            if tensorboard_directory is None:
                # The parameter was not given, set the directory to the default value:
                tensorboard_directory = self._DEFAULT_TENSORBOARD_DIRECTORY.replace(
                    "{{project}}", self._context.project
                )
            # Build the full run name:
            full_run_name = "{}_{}".format(run_name, self._context.uid)
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
            full_run_name = "{}_{}".format(run_name, index)

        # Create the output path:
        output_path = os.path.join(tensorboard_directory, full_run_name)
        os.makedirs(output_path)

        return full_run_name, output_path
