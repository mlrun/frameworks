from typing import Union, List, Dict
import mlrun
from frameworks._common.loggers import MLRunLogger, TrackableType
from frameworks.keras.callbacks.logging_callback import LoggingCallback
from frameworks.keras.utilities.model_handler import KerasModelHandler


class MLRunLoggingCallback(LoggingCallback):
    """
    Callback for logging data during training / validation via mlrun's context. Each tracked hyperparameter and metrics
    results will be logged per epoch and at the end of the run the model will be saved and logged as well. Some plots
    will be available as well. To summerize, the available data in mlrun will be:

    * For each epoch:

      * Tracking table: epoch, static hyperparameters, dynamic hyperparameters, training metrics, validation metrics.
      * Per iteration (batch) chart artifacts for the training and validation metrics.

    * At the end of the run:

      * Per epoch chart artifacts for the validation summaries and dynamic hyperparameters.
      * Model is logged with all of the files and artifacts.

    All the collected data will be available in this callback post the training / validation process and can be accessed
    via the 'training_results', 'validation_results', 'static_hyperparameters', 'dynamic_hyperparameters' and
    'summaries' properties.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        dynamic_hyperparameters: Dict[str, List[Union[str, int]]] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, List[Union[str, int]]]
        ] = None,
        per_iteration_logging: int = 1,
    ):
        """
        Initialize an mlrun logging callback with the given hyperparameters and logging configurations.

        :param context:                 The mlrun context to log with.
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
        super(MLRunLoggingCallback, self).__init__(
            dynamic_hyperparameters=dynamic_hyperparameters,
            static_hyperparameters=static_hyperparameters,
            per_iteration_logging=per_iteration_logging,
        )

        # Replace the logger with an MLRunLogger:
        del self._logger
        self._logger = MLRunLogger(context=context)

    def on_train_end(self, logs: dict = None):
        # TODO: Need to finish up the model handler
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, TrackableType] = None):
        """
        Called at the end of an epoch, logging the dynamic hyperparameters and results of this epoch via the stored
        context.

        :param epoch: Integer, index of epoch.
        :param logs:  Dict, metric results for this training epoch, and for the validation epoch if validation is
                      performed. Validation result keys are prefixed with `val_`. For training epoch, the values of the
                      `Model`'s metrics are returned. Example : `{'loss': 0.2, 'acc': 0.7}`.
        """
        super(MLRunLoggingCallback, self).on_epoch_end(epoch=epoch)

        # Create child context to hold the current epoch's results:
        self._logger.log_epoch_to_context(epoch=epoch)
