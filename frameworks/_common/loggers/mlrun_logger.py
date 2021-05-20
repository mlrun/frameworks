from typing import Dict, List, Union
import numpy as np
import mlrun
from mlrun import MLClientCtx
from mlrun.artifacts import Artifact, ChartArtifact
from frameworks._common.loggers.logger import Logger


# All trackable values types:
TrackableType = Union[str, bool, float, int]


class MLRunLogger(Logger):
    """
    MLRun logger is logging the information collected during training / evaluation of the base logger and logging it to
    MLRun using a MLRun context. The logging includes:

    * For each epoch:

      * Tracking table: epoch, static hyperparameters, dynamic hyperparameters, training metrics, validation metrics.
      * Per iteration (batch) chart artifacts for the training and validation metrics.

    * At the end of the run:

      * Per epoch chart artifacts for the validation summaries and dynamic hyperparameters.
      * Model is logged with all of the files and artifacts.
    """

    def __init__(self, context: MLClientCtx):
        """
        Initialize the MLRun logging interface to work with the given context.
        :param context: MLRun context to log to.
        """
        super(MLRunLogger, self).__init__()

        # Store the context:
        self._context = context

        # Prepare the artifacts collection:
        self._artifacts = {}  # type: Dict[str, Artifact]

    def log_epoch_to_context(
        self,
        epoch: int,
    ):
        """
        Log the last epoch as a child context of the main context. The last epoch information recorded in the given
        tracking dictionaries will be logged, meaning the epoch index will not be taken from the given 'epoch'
        parameter, but the '-1' index will be used in all of the dictionaries. Each epoch will log the following
        information:
        * Results table:
            - Static hyperparameters.
            - Dynamic hyperparameters.
            - Last iteration recorded training results for loss and metrics.
            - Validation results summaries for loss and metrics.
        * Plot artifacts:
            - A chart for each of the metrics iteration results in training.
            - A chart for each of the metrics iteration results in validation.
        :param epoch: The epoch number that has just ended.
        """
        # Create child context to hold the current epoch's results:
        child_context = self._context.get_child_context()

        # Set the current iteration number according to the epoch number:
        child_context._iteration = epoch

        # Log the collected hyperparameters and values as results to the epoch's child context:
        for static_parameter, value in self._static_hyperparameters.items():
            child_context.log_result(static_parameter, value)
        for dynamic_parameter, values in self._dynamic_hyperparameters.items():
            child_context.log_result(dynamic_parameter, values[-1])
        for metric, epochs in self._training_results.items():
            child_context.log_result("Training {}".format(metric), epochs[-1][-1])
        for metric, results in self._summaries.items():
            child_context.log_result("Validation {}".format(metric), results[-1])

        # Update the last epoch to the main context:
        self._context._results = child_context.results

        # Log the epochs metrics results as chart artifacts:
        for metrics_prefix, metrics_dictionary in zip(
            ["Train", "Validation"], [self._training_results, self._validation_results]
        ):
            for metric_name, metric_epochs in metrics_dictionary.items():
                # Create the chart artifact:
                chart_name = "{}_{}_results_epoch_{}".format(
                    metrics_prefix, metric_name, len(metric_epochs)
                )
                chart_artifact = ChartArtifact(
                    key="{}.html".format(chart_name),
                    header=["iteration", metric_name],
                    data=np.array(
                        [list(np.arange(len(metric_epochs[-1]))), metric_epochs[-1]]
                    ).transpose(),
                )
                # Log the artifact:
                child_context.log_artifact(
                    chart_artifact,
                    local_path=chart_artifact.key,
                    artifact_path=child_context.artifact_path,
                )
                # Collect it for later adding it to the model logging as extra data:
                self._artifacts[chart_name] = chart_artifact

        # Commit and commit children for MLRun flag bug:
        self._context.update_child_iterations(commit_children=True)
        self._context.commit()

    def log_run(self):
        pass
