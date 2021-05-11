from typing import Union, List, Dict, Tuple
import os
import mlrun
from mlrun.artifacts import ChartArtifact
from frameworks.keras.callbacks.logging_callback import LoggingCallback, TrackableType


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

        # Store the context:
        self._context = context

    def on_train_end(self, logs: dict = None):
        # Save the model:
        # self.model.save(os.path.join(artifact_path, self.model.name))
        # model_directory_artifact = self._context.log_artifact(
        #     "model-directory",
        #     local_path=self.model.name,
        #     artifact_path=artifact_path,
        #     db_key=False,
        # )
        self.model.save("{}.h5".format(self.model.name))

        # Save weights
        self.model.save_weights("{}-weights.h5".format(self.model.name))
        weights_artifact = self._context.log_artifact(
            "{}-weights".format(self.model.name),
            local_path="{}-weights.h5".format(self.model.name),
            artifact_path=self._context.artifact_path,
            db_key=False,
        )

        # Produce training chart artifact
        chart_name = "summary.html"
        chart_artifact = ChartArtifact(chart_name)
        chart_artifact.header = (
            ["epoch"]
            + list(self._static_hyperparameters.keys())
            + list(self._dynamic_hyperparameters.keys())
            + list(self._summaries.keys())
        )
        for i in range(self._epochs + 1):
            row = [i]
            for value in self._static_hyperparameters.values():
                row.append(value)
            for epoch_values in self._dynamic_hyperparameters.values():
                row.append(epoch_values[i])
            for epoch_values in self._summaries.values():
                if i == 0:
                    row.append("-")
                else:
                    row.append(epoch_values[i - 1])
            chart_artifact.add_row(row=row)
        summary_artifact = self._context.log_artifact(
            chart_artifact,
            local_path=chart_name,
            artifact_path=self._context.artifact_path,
        )

        # Log the model as a `model` artifact in MLRun:
        self._context.log_model(
            "model",
            artifact_path=self._context.artifact_path,
            model_file="{}.h5".format(self.model.name),
            labels={"framework": "tensorflow"},
            metrics=self._context.results,
            extra_data={
                "training-summary": summary_artifact,
                "model-architecture.json": bytes(self.model.to_json(), encoding="utf8"),
                "model-weights.h5": weights_artifact,
            },
        )
        self._context.commit()

    def on_epoch_end(self, epoch: int, logs: Dict[str, TrackableType] = None):
        """
        Called at the end of an epoch.
        :param epoch: Integer, index of epoch.
        :param logs:  Dict, metric results for this training epoch, and for the validation epoch if validation is
                      performed. Validation result keys are prefixed with `val_`. For training epoch, the values of the
                      `Model`'s metrics are returned. Example : `{'loss': 0.2, 'acc': 0.7}`.
        """
        super(MLRunLoggingCallback, self).on_epoch_end(epoch=epoch)

        # Create child context to hold the current epoch's results:
        child_ctx = self._context.get_child_context()

        # Set the current iteration number according to the epoch number:
        child_ctx._iteration = self._epochs

        # Go over the static hyperparameters and log them to the context:
        for parameter, value in self._static_hyperparameters.items():
            child_ctx.log_result(parameter, value)

        # Go over the dynamic hyperparameters and log them to the context:
        for parameter, epochs in self._dynamic_hyperparameters.items():
            child_ctx.log_result(parameter, epochs[-1])

        # Go over the summaries and log them to the context:
        for metric, epochs in self._summaries.items():
            child_ctx.log_result(metric, epochs[-1])

        # Commit and commit children for MLRun flag bug:
        self._context.update_child_iterations(commit_children=True)
        self._context.commit()
