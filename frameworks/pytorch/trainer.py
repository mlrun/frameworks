from typing import Tuple, List
import sys
from tabulate import tabulate
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from mlrun.execution import MLClientCtx
from frameworks.base.trainer import Trainer
from callbacks.callback import Callback, MetricFunctionType, MetricValueType
from callbacks_handler import CallbacksHandler


class PyTorchTrainer(Trainer):
    """
    An interface for a pytorch model trainer, supporting the package's callbacks and automatic logging.
    """

    def __init__(
        self,
        model: Module,
        training_set: DataLoader,
        validation_set: DataLoader,
        loss_function: Module,
        optimizer: Optimizer,
        metric_functions: List[MetricFunctionType] = None,
        scheduler=None,
        epochs: int = 1,
        training_iterations: int = None,
        validation_iterations: int = None,
    ):
        """
        Initialize a trainer for a given experiment objects.
        :param model:                 The model to train.
        :param training_set:          A data loader for the training process.
        :param validation_set:        A data loader for the validation process.
        :param loss_function:         The loss function to use during training.
        :param optimizer:             The optimizer to use during the training.
        :param metric_functions:      The metrics to use on training and validation.
        :param scheduler:             Scheduler to use on the optimizer at the end of each epoch. The scheduler must
                                      have a 'step' method with no input.
        :param epochs:                Amount of epochs to perform. Defaulted to a single epoch.
        :param training_iterations:   Amount of iterations (batches) to perform on each epoch's training. If 'None' the
                                      entire training set will be used.
        :param validation_iterations: Amount of iterations (batches) to perform on each epoch's validation. If 'None'
                                      the entire validation set will be used.
        """
        # Store the configurations:
        self._model = model
        self._training_set = training_set
        self._validation_set = validation_set
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._metric_functions = (
            metric_functions if metric_functions is not None else []
        )
        self._scheduler = scheduler
        self._epochs = epochs
        self._training_iterations = (
            len(training_set)
            if training_iterations is None or training_iterations > len(training_set)
            else training_iterations
        )
        self._validation_iterations = (
            len(validation_set)
            if validation_iterations is None
            or validation_iterations > len(validation_set)
            else validation_iterations
        )

    def run(self, callbacks: List[Callback] = None):
        """
        Run the trainer training process on his initialized configuration.
        :param callbacks: The callbacks to use on this run.
        """
        # Initialize a callbacks handler:
        callbacks_handler = CallbacksHandler(
            callbacks=callbacks if callbacks is not None else []
        )

        # Setup the callbacks functions:
        callbacks_handler.on_setup(
            model=self._model,
            training_set=self._training_set,
            validation_set=self._validation_set,
            loss_function=self._loss_function,
            optimizer=self._optimizer,
            metric_functions=self._metric_functions,
            scheduler=self._scheduler,
        )

        # Beginning of run callbacks:
        callbacks_handler.on_run_begin()

        # Start the epochs:
        for epoch in range(self._epochs):
            # Beginning of a epoch callbacks:
            callbacks_handler.on_epoch_begin(epoch=epoch)
            print(
                "Epoch {}/{}:".format(
                    str(epoch + 1).rjust(len(str(self._epochs))), self._epochs
                )
            )

            # Train:
            callbacks_handler.on_train_begin()
            self._train(callbacks_handler=callbacks_handler)
            if not callbacks_handler.on_train_end():
                break

            # Validate:
            callbacks_handler.on_validation_begin()
            loss_value, metric_values = self._validate(
                callbacks_handler=callbacks_handler
            )
            self._print_results(loss_value=loss_value, metric_values=metric_values)
            if not callbacks_handler.on_validation_end(
                loss_value=loss_value, metric_values=metric_values
            ):
                break

            # Step scheduler:
            if self._scheduler:
                callbacks_handler.on_scheduler_step_begin()
                self._scheduler.step()
                callbacks_handler.on_scheduler_step_end()

            # End of a epoch callbacks:
            if not callbacks_handler.on_epoch_end(epoch=epoch):
                break
            print()

        # End of run callbacks:
        callbacks_handler.on_run_end()

    def auto_log(self, context: MLClientCtx):
        """
        Run training with automatic logging to mlrun's context and tensorboard.
        :param context: The context to use for the logs.
        """
        raise NotImplementedError

    def _metrics(self, y_pred: Tensor, y_true: Tensor) -> List[float]:
        """
        Call all the metrics on the given batch's truth and prediction output.
        :param y_pred: The batch's truth value.
        :param y_true: The model's output for the related input.
        :return: A list with each metric result.
        """
        accuracies = []
        for metric_function in self._metric_functions:
            accuracies.append(metric_function(y_pred, y_true))
        return accuracies

    def _train(self, callbacks_handler: CallbacksHandler):
        """
        Initiate a single epoch training.
        :param callbacks_handler: Callbacks handler to use.
        """
        # Set model to train mode:
        self._model.train()

        # Start the training:
        progress_bar = tqdm(
            iterable=enumerate(self._training_set),
            bar_format="{desc}:   {percentage:3.0f}%"
            " |{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            desc="Training",
            postfix={"Loss": "?"},
            unit="Batch",
            total=self._training_iterations,
            ascii=False,
            file=sys.stdout,
        )
        for batch, (x, y_true) in progress_bar:
            if batch == self._training_iterations:
                break
            # Beginning of a batch callbacks:
            callbacks_handler.on_train_batch_begin(batch=batch, x=x, y_true=y_true)
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
                y_true = y_true.cuda(non_blocking=True)

            # Infer the input:
            y_pred = self._model(x)

            # Calculate loss:
            callbacks_handler.on_train_loss_begin()
            loss_value = self._loss_function(y_pred, y_true)
            progress_bar.set_postfix(Loss=float(loss_value), refresh=False)
            callbacks_handler.on_train_loss_end(loss_value=loss_value)

            # Measure accuracies:
            callbacks_handler.on_train_metrics_begin()
            metric_values = self._metrics(y_pred=y_pred, y_true=y_true)
            callbacks_handler.on_train_metrics_end(metric_values=metric_values)

            # Perform backward propagation:
            callbacks_handler.on_backward_begin()
            loss_value.backward()
            callbacks_handler.on_backward_end()

            # Step optimizer:
            callbacks_handler.on_optimizer_step_begin()
            self._optimizer.step()
            self._optimizer.zero_grad()
            callbacks_handler.on_optimizer_step_end()

            # End of batch callbacks:
            if not callbacks_handler.on_train_batch_end(
                batch=batch, x=x, y_true=y_true, y_pred=y_pred
            ):
                break

    def _validate(
        self, callbacks_handler: CallbacksHandler
    ) -> Tuple[MetricValueType, List[MetricValueType]]:
        """
        Initiate a single epoch validation.
        :param callbacks_handler: Callbacks handler to use.
        :return: A tuple of the validation summary:
                 [0] = Validation loss value summary.
                 [1] = A list of metrics summaries.
        """
        # Set model to evaluate mode:
        self._model.eval()

        # Start the validation:
        losses = []
        metrics = []
        progress_bar = tqdm(
            iterable=enumerate(self._validation_set),
            bar_format="{desc}: {percentage:3.0f}%"
            " |{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            desc="Validating",
            postfix={"Loss": "?"},
            unit="Batch",
            total=self._validation_iterations,
            ascii=False,
            file=sys.stdout,
        )
        with torch.no_grad():
            for batch, (x, y_true) in progress_bar:
                if batch == self._validation_iterations:
                    break
                # Beginning of a batch callbacks:
                callbacks_handler.on_validation_batch_begin(
                    batch=batch, x=x, y_true=y_true
                )
                if torch.cuda.is_available():
                    x = x.cuda(non_blocking=True)
                    y_true = y_true.cuda(non_blocking=True)

                # Infer the input:
                y_pred = self._model(x)

                # Calculate loss:
                callbacks_handler.on_validation_loss_begin()
                loss_value = self._loss_function(y_pred, y_true)
                progress_bar.set_postfix(Loss=float(loss_value), refresh=False)
                callbacks_handler.on_validation_loss_end(loss_value=loss_value)

                # Measure accuracies:
                callbacks_handler.on_validation_metrics_begin()
                metric_values = self._metrics(y_pred=y_pred, y_true=y_true)
                callbacks_handler.on_validation_metrics_end(metric_values=metric_values)

                # Collect results:
                losses.append(loss_value)
                metrics.append(metric_values)

                # End of batch callbacks:
                if not callbacks_handler.on_validation_batch_end(
                    batch=batch, x=x, y_true=y_true, y_pred=y_pred
                ):
                    break

        # Calculate the final average of the loss and accuracy values:
        loss_value = sum(losses) / len(losses)
        metric_values = [(sum(metric) / len(metric)) for metric in metrics]
        return loss_value, metric_values

    def _print_results(self, loss_value: Tensor, metric_values: List[float]):
        """
        Print the given result between each epoch.
        :param loss_value:    The loss result to print.
        :param metric_values: The metrics result to print.
        """
        table = [[self._loss_function.__class__.__name__, float(loss_value)]]
        for metric_function, metric_value in zip(self._metric_functions, metric_values):
            if isinstance(metric_function, Module):
                metric_name = metric_function.__class__.__name__
            else:
                metric_name = metric_function.__name__
            table.append([metric_name, metric_value])
        print(
            "\nSummary:\n"
            + tabulate(table, headers=["Metrics", "Values"], tablefmt="pretty")
        )
