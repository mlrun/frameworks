from typing import List, Tuple, Dict, Union, Any
import datetime
import os
import tensorflow as tf
from tensorflow import Tensor, Variable
from tensorflow.python.ops import summary_ops_v2
from tensorboard.plugins.hparams import api as hp_api
from tensorboard.plugins.hparams import api_pb2 as hp_api_pb2
from tensorboard.plugins.hparams import summary as hp_summary
from tensorflow import keras
from frameworks.keras.callbacks.logging_callback import LoggingCallback, TrackableType
import mlrun


class TensorboardLoggingCallback(LoggingCallback):
    """
    Callback for logging data during training / evaluation to tensorboard. Each tracked metrics results will be logged
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
        dynamic_hyperparameters: Dict[str, List[Union[str, int]]] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, List[Union[str, int]]]
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
                                        hyperparameter chosen names and the values are a key chain. A key chain is a
                                        list of keys and indices to know how to access the needed hyperparameter from
                                        the compiled model. For example, to track the 'lr' attribute of an optimizer,
                                        one should pass:
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
        # [Weight: str] -> [value: Variable]
        self._weights = {}  # type: Dict[str, Variable]

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
        self._file_writer = tf.summary.create_file_writer(self._output_path)
        self._file_writer.set_as_default()

        # Initialize flags:
        self._is_training = False
        self._logged_model = False
        self._logged_hyperparameters = False

    @property
    def weights(self) -> Dict[str, Tensor]:
        """
        Get the weights tensors tracked. The weights will be stored in a dictionary where each key is the weight's name
        and the value is the weight's tensor.
        :return: The weights.
        """
        weights = {}
        for weight_name, weight_variable in self._weights.items():
            weights[weight_name] = tf.convert_to_tensor(weight_variable)
        return weights

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

    def on_train_begin(self, logs: dict = None):
        # The callback is on a 'fit' method - training:
        self._is_training = True

        # Prepare to run:
        self._setup_run()

    def on_train_end(self, logs: dict = None):
        super(TensorboardLoggingCallback, self).on_train_end()
        self._recap_run()

    def on_test_begin(self, logs: dict = None):
        # If this callback is part of evaluation and not training, need to check if the run was setup:
        if not self._run_set_up:
            self._setup_run()

    def on_test_end(self, logs: Dict[str, Any] = None):
        super(TensorboardLoggingCallback, self).on_test_end(logs=logs)

        # Check if needed to end the run:
        if not self._is_training:
            self._recap_run()

    def on_epoch_end(self, epoch: int, logs: Dict[str, TrackableType] = None):
        # Update the dynamic hyperparameters
        super(TensorboardLoggingCallback, self).on_epoch_end(epoch=epoch)

        # Add this epoch loss and metrics averages to their graphs:
        self._log_per_epoch_scalars(
            results=self._summaries, section=self._Sections.SUMMARY, epoch=self._epochs
        )

        # Add this epoch dynamic hyperparameters values to their graphs:
        if self._dynamic_hyperparameters:
            self._log_per_epoch_scalars(
                results=self._dynamic_hyperparameters,
                section=self._Sections.HYPERPARAMETERS,
                epoch=self._epochs,
            )

        # Add weight histograms and statistics for all the tracked weights:
        if self._tracked_weights:
            self._log_weights(epoch=epoch)

        # Make sure all values were written to the directory logs:
        self._file_writer.flush()

    def on_train_batch_begin(self, batch: int, logs: Dict[str, TrackableType] = None):
        super(TensorboardLoggingCallback, self).on_train_batch_begin(
            batch=batch, logs=logs
        )
        if not self._logged_model:
            summary_ops_v2.trace_on(graph=True, profiler=False)

    def on_train_batch_end(self, batch: int, logs: dict = None):
        # Log the batch's results:
        super(TensorboardLoggingCallback, self).on_train_batch_end(
            batch=batch, logs=logs
        )

        # Add this batch loss and metrics results to their graphs:
        self._log_per_batch_scalars(
            results=self._training_results,
            section=self._Sections.TRAINING,
            iteration=self._train_iterations,
        )

        # Check if needed to log model:
        if not self._logged_model:
            self._log_model(
                epoch=self._epochs, batch=batch, iteration=self._train_iterations
            )

        # Check if needed to log hyperparameters:
        if (
            not (
                len(self._static_hyperparameters) == 0
                and len(self._dynamic_hyperparameters) == 0
            )
            and not self._logged_hyperparameters
        ):
            self._log_hyperparameters(logs=logs)

    def on_test_batch_begin(self, batch: int, logs: dict = None):
        super(TensorboardLoggingCallback, self).on_test_batch_begin(
            batch=batch, logs=logs
        )

        if not self._logged_model:
            summary_ops_v2.trace_on(graph=True, profiler=False)

    def on_test_batch_end(self, batch: int, logs: dict = None):
        # Log the batch's results:
        super(TensorboardLoggingCallback, self).on_test_batch_end(
            batch=batch, logs=logs
        )

        # Add this batch loss and metrics results to their graphs:
        self._log_per_batch_scalars(
            results=self._validation_results,
            section=self._Sections.VALIDATION,
            iteration=self._validation_iterations,
        )

        # Check if needed to log model:
        if not self._logged_model:
            self._log_model(
                epoch=self._epochs, batch=batch, iteration=self._validation_iterations
            )

        # Check if needed to log hyperparameters:
        if (
            not (
                len(self._static_hyperparameters) == 0
                and len(self._dynamic_hyperparameters) == 0
            )
            and not self._logged_hyperparameters
        ):
            self._log_hyperparameters(logs=logs)

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

    def _recap_run(self):
        # Close the hyperparameters writing:
        if not (
            len(self._static_hyperparameters) == 0
            and len(self._dynamic_hyperparameters) == 0
        ):
            with self._file_writer.as_default():
                pb = hp_summary.session_end_pb(hp_api_pb2.STATUS_SUCCESS)
                raw_pb = pb.SerializeToString()
                tf.compat.v2.summary.experimental.write_raw_pb(raw_pb, step=0)

        # Close the file writer:
        self._file_writer.flush()
        self._file_writer.close()

    def _setup_run(self):
        """
        After the trainer / evaluator run begins, this method will be called to setup the results, hyperparameters
        and weights dictionaries for logging.
        """
        super(TensorboardLoggingCallback, self)._setup_run()

        # Collect the weights for drawing histograms according to the stored configuration:
        if self._tracked_weights:
            for layer in self.model.layers:
                collect = False
                if self._tracked_weights is True:  # Collect all weights
                    collect = True
                else:
                    for tag in self._tracked_weights:  # Collect by given name
                        if tag in layer.name:
                            collect = True
                            break
                if collect:
                    for weight_variable in layer.weights:
                        self._weights[weight_variable.name] = weight_variable
                        self._weights_mean[weight_variable.name] = [
                            float(tf.reduce_mean(weight_variable))
                        ]
                        self._weights_std[weight_variable.name] = [
                            float(tf.math.reduce_std(weight_variable))
                        ]
            # Log the initial state of the collected weights:
            self._log_weights(epoch=0)

    def _log_hyperparameters(self, logs: dict):
        self._logged_hyperparameters = True

        # Prepare the static hyperparameters values:
        non_graph_parameters = {}
        hp_param_list = []
        for parameter, value in self._static_hyperparameters.items():
            non_graph_parameters[parameter] = value
            hp_param_list.append(hp_api.HParam(parameter))

        # Prepare the summaries values and the dynamic hyperparameters values (both registered as metrics):
        graph_parameters = {}
        hp_metric_list = []
        for metric in logs:
            metric_name = "{}/{}".format(
                self._Sections.SUMMARY, self._get_metric_name(metric_name_in_log=metric)
            )
            graph_parameters[metric_name] = 0.0
            hp_metric_list.append(hp_api.Metric(metric_name))
        for parameter, epochs in self._dynamic_hyperparameters.items():
            parameter_name = "{}/{}".format(self._Sections.HYPERPARAMETERS, parameter)
            graph_parameters[parameter_name] = epochs[-1]
            hp_metric_list.append(hp_api.Metric(parameter_name))

        # Write the hyperparameters and summaries to the table:
        with self._file_writer.as_default():
            hp_api.hparams_config(hparams=hp_param_list, metrics=hp_metric_list)
            hp_api.hparams(non_graph_parameters, trial_id=self._run_name)

        # Add initial dynamic hyperparameters values (epoch 0) to their graphs:
        self._log_per_epoch_scalars(
            results=self._dynamic_hyperparameters,
            section=self._Sections.HYPERPARAMETERS,
            epoch=0,
        )

    def _log_weights(self, epoch: int):
        with self._file_writer.as_default():
            for weight_name, weight_variable in self._weights.items():
                # Draw histogram:
                tf.summary.histogram(
                    name="{}/{}".format(self._Sections.WEIGHTS, weight_name),
                    data=weight_variable,
                    step=epoch,
                )
                # Add Mean value:
                tf.summary.scalar(
                    name="{}/{}:{}".format(
                        self._Sections.WEIGHTS, weight_name, self._StatisticsTypes.MEAN
                    ),
                    data=self._weights_mean[weight_name][-1],
                    step=epoch,
                )
                # Add Standard-deviation value:
                tf.summary.scalar(
                    name="{}/{}:{}".format(
                        self._Sections.WEIGHTS, weight_name, self._StatisticsTypes.STD
                    ),
                    data=self._weights_std[weight_name][-1],
                    step=epoch,
                )

    def _log_per_batch_scalars(
        self, results: Dict[str, List[List[float]]], section: str, iteration: int
    ):
        with self._file_writer.as_default():
            for parameter, epochs in results.items():
                tf.summary.scalar(
                    name="{}/{}".format(section, parameter),
                    data=epochs[-1][-1],
                    step=iteration,
                )

    def _log_per_epoch_scalars(
        self, results: Dict[str, List[float]], section: str, epoch: int
    ):
        with self._file_writer.as_default():
            for parameter, epochs in results.items():
                tf.summary.scalar(
                    name="{}/{}".format(section, parameter),
                    data=epochs[-1],
                    step=epoch,
                )

    def _log_model(self, epoch: int, batch: int, iteration: int):
        self._logged_model = True
        with self._file_writer.as_default():
            with summary_ops_v2.always_record_summaries():
                summary_ops_v2.trace_export(
                    name="epoch_{}_batch_{}".format(epoch, batch), step=iteration
                )
                summary_ops_v2.keras_model(
                    name=self.model.name, data=self.model, step=iteration
                )
