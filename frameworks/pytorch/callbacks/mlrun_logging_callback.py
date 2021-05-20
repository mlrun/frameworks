from typing import Union, List, Dict, Tuple
import numpy as np
import mlrun
from mlrun.artifacts import ChartArtifact, Artifact
from frameworks._common.loggers import MLRunLogger
from frameworks.pytorch.callbacks.logging_callback import LoggingCallback, TrackableType
from frameworks.pytorch.utilities.model_handler import PyTorchModelHandler


class MLRunLoggingCallback(LoggingCallback):
    """
    Callback for logging data during training / validation via mlrun's context. Each tracked hyperparameter and metrics
    results will be logged per epoch and at the end of the run the model will be saved and logged as well. Some plots
    will be available as well.

    To summerize, the available data in mlrun will be:
        * Plot artifacts:
            - Summaries for loss, metrics per epoch.
            - Dynamic hyperparameters per epoch.
        * Results table:
            - Summaries for loss, metrics per epoch.
            - Dynamic hyperparameters per epoch.
        * Model:
            - Model files stored in the project's directory.
            - A model log page linked to all of his artifacts.

    All the collected data will be available in this callback post the training / validation process and can be accessed
    via the 'training_results', 'validation_results', 'static_hyperparameters', 'dynamic_hyperparameters' and
    'summaries' properties.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        dynamic_hyperparameters: Dict[str, Tuple[str, List[Union[str, int]]]] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, Tuple[str, List[Union[str, int]]]]
        ] = None,
        per_iteration_logging: int = 1,
    ):
        """
        Initialize an mlrun logging callback with the given hyperparameters and logging configurations.
        :param context:                 The mlrun context to log with.
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
        super(MLRunLoggingCallback, self).__init__(
            dynamic_hyperparameters=dynamic_hyperparameters,
            static_hyperparameters=static_hyperparameters,
            per_iteration_logging=per_iteration_logging,
        )

        # Replace the logger with an MLRunLogger:
        del self._logger
        self._logger = MLRunLogger(context=context)

    def on_run_end(self):
        # TODO: Need to finish up the model holder
        pass

    def on_epoch_end(self, epoch: int):
        """
        Before the trainer given epoch ends, this method will be called to log the dynamic hyperparameters and results
        of this epoch via the stored context.
        :param epoch: The epoch that has just ended.
        """
        super(MLRunLoggingCallback, self).on_epoch_end(epoch=epoch)

        # Create child context to hold the current epoch's results:
        self._logger.log_epoch_to_context(epoch=epoch)
