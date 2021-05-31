from typing import Union, Tuple, List, Dict
import sys

from tabulate import tabulate
from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer

from mlrun.execution import MLClientCtx
from frameworks.pytorch.callbacks import (
    Callback,
    MetricFunctionType,
    MetricValueType,
    TrackableType,
    HyperparametersKeys,
    MLRunLoggingCallback,
    TensorboardLoggingCallback,
)
from frameworks.pytorch.utilities.callbacks_handler import CallbacksHandler


class Trainer:
    """
    An interface for a pytorch model trainer, supporting the package's loggers and automatic logging.
    """

    def __init__(
        self,
        model: Module,
        training_set: Union[Dataset, DataLoader],
        loss_function: Module,
        optimizer: Optimizer,
        validation_set: Union[Dataset, DataLoader] = None,
        metric_functions: List[MetricFunctionType] = None,
        scheduler=None,
        epochs: int = 1,
        training_iterations: int = None,
        validation_iterations: int = None,
    ):
        """
        Initialize a trainer for a given experiment objects.
        :param model:                 The model to train.
        :param training_set:          A dataset or data loader for the training process. If a dataset is given, a
                                      defaulted data loader would be used for training.
        :param loss_function:         The loss function to use during training.
        :param optimizer:             The optimizer to use during the training.
        :param validation_set:        A dataset or data loader for the validation process. If a dataset is given, a
                                      defaulted data loader would be used for training.
        :param metric_functions:      The metrics to use on training and validation.
        :param scheduler:             Scheduler to use on the optimizer at the end of each epoch. The scheduler must
                                      have a 'step' method with no input.
        :param epochs:                Amount of epochs to perform. Defaulted to a single epoch.
        :param training_iterations:   Amount of iterations (batches) to perform on each epoch's training. If 'None' the
                                      entire training set will be used.
        :param validation_iterations: Amount of iterations (batches) to perform on each epoch's validation. If 'None'
                                      the entire validation set will be used.
        """
        # TODO: Add and align features of keras to PyTorch (like validation frequency for example).
        # Store the configurations:
        self._model = model
        self._training_set = (
            training_set
            if isinstance(training_set, DataLoader)
            else DataLoader(training_set)
        )
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._validation_set = (
            validation_set
            if isinstance(validation_set, DataLoader)
            else DataLoader(validation_set)
        )
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

    def run(
        self,
        callbacks: List[Callback] = None,
        use_cuda: bool = True,
        use_horovod: bool = False,
    ):
        """
        Run the trainer training process on his initialized configuration.

        :param callbacks:   The loggers to use on this run.
        :param use_cuda:    Whether or not to use cuda. Only relevant if cuda is available. Defaulted to True.
        :param use_horovod: Whether or not to use horovod - a distributed training framework. Defaulted to False.
        """
        # Setup horovod:
        hvd = None
        if use_horovod:
            import horovod.torch as hvd

            hvd.init()

        # Setup cuda:
        if use_cuda and torch.cuda.is_available():
            if use_horovod:
                torch.cuda.set_device(hvd.local_rank())
            self._model.cuda()

        # Initialize a callbacks handler:
        callbacks = callbacks if callbacks is not None else []
        if use_horovod:
            callbacks_handler = CallbacksHandler(
                callbacks=[
                    callback
                    for callback in callbacks
                    if callback.on_horovod_check(rank=hvd.local_rank())
                ]
            )
        else:
            callbacks_handler = CallbacksHandler(callbacks=callbacks)

        # Prepare horovod for the run if needed:
        if use_horovod:
            # Partition dataset among workers using DistributedSampler:
            self._training_set.sampler = DistributedSampler(
                self._training_set.dataset, num_replicas=hvd.size(), rank=hvd.rank()
            )
            if self._validation_set:
                self._validation_set.sampler = DistributedSampler(
                    self._validation_set.dataset,
                    num_replicas=hvd.size(),
                    rank=hvd.rank(),
                )
            # Add Horovod Distributed Optimizer:
            self._optimizer = hvd.DistributedOptimizer(
                self._optimizer, named_parameters=self._model.named_parameters()
            )
            # Broadcast parameters from rank 0 to all other processes:
            hvd.broadcast_parameters(self._model.state_dict(), root_rank=0)

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
            # Beginning of a epoch loggers:
            callbacks_handler.on_epoch_begin(epoch=epoch)
            print(
                "Epoch {}/{}:".format(
                    str(epoch + 1).rjust(len(str(self._epochs))), self._epochs
                )
            )

            # Train:
            callbacks_handler.on_train_begin()
            self._train(callbacks_handler=callbacks_handler, use_cuda=use_cuda)
            if not callbacks_handler.on_train_end():
                break

            # Validate:
            if self._validation_set is not None:
                callbacks_handler.on_validation_begin()
                loss_value, metric_values = self._validate(
                    callbacks_handler=callbacks_handler, use_cuda=use_cuda
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

            # End of a epoch loggers:
            if not callbacks_handler.on_epoch_end(epoch=epoch):
                break
            print()

        # End of run loggers:
        callbacks_handler.on_run_end()

    def get_auto_logging_callbacks(
        self,
        context: MLClientCtx,
        custom_objects: Dict[Union[str, List[str]], str] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, Tuple[str, List[Union[str, int]]]]
        ] = None,
        log_learning_rate: bool = True,
    ) -> List[Callback]:
        """
        Get automatic logging callbacks to both MLRun's context and Tensorboard. For further features of logging to both
        MLRun and Tensorboard, see 'pytorch.callbacks.MLRunLoggingCallback' and
        'pytorch.callbacks.TensorboardLoggingCallback'.

        :param context:                The context to use for the logs.
        :param custom_objects:         Custom objects the model is using. Expecting a dictionary with the classes names
                                       to import as keys (if multiple classes needed to be imported from the same py
                                       file a list can be given) and the python file from where to import them as their
                                       values. The model class itself must be specified in order to properly save it for
                                       later being loaded with a handler. For example:
                                       {
                                           "class_name": "/path/to/model.py",
                                           ["layer1", "layer2"]: "/path/to/custom_layers.py"
                                       }
        :param static_hyperparameters: A dictionary of static hyperparameters to note in the logs. The parameter expects
                                       a dictionary where the keys are the hyperparameter chosen names and the values
                                       are tuples of string (from HyperparametersKeys) and a key chain - a list of keys
                                       and indices to know how to access the needed hyperparameter from the object. For
                                       example, to track the 'epsilon' attribute of an optimizer and the 'epochs' of an
                                       experiment run, one should pass:
                                       {
                                           "epsilon": ["optimizer", "epsilon"],
                                           "epochs": 7
                                       }
                                       Defaulted to the following static hyperparameters: batch size, epochs, training
                                       and validation (if given validation set) iterations.
        :param log_learning_rate:      Whether or not to log the learning rate of the given optimizer. Notice, if the
                                       learning rate is not accessible via the common key chain
                                       (param_group 0 -> 'lr'/'learning_rate') it won't be tracked. To track it you
                                       should create the callbacks and use the 'run' method. Defaulted to True.
        """
        # Define the static hyperparameters:
        if static_hyperparameters is None:
            static_hyperparameters = {
                "Batch Size": self._training_set.batch_size,
                "Epochs": self._epochs,
                "Training Iterations": self._training_iterations,
            }
            if self._validation_set is not None:
                static_hyperparameters[
                    "Validation Iterations"
                ] = self._validation_iterations

        # Define the dynamic hyperparameters:
        dynamic_hyperparameters = {}
        if log_learning_rate:
            learning_rate = self._get_learning_rate()
            if learning_rate is not None:
                dynamic_hyperparameters["Learning Rate"] = learning_rate

        # Initialize and return the callbacks:
        return [
            MLRunLoggingCallback(
                context=context,
                custom_objects=custom_objects,
                static_hyperparameters=static_hyperparameters,
                dynamic_hyperparameters=dynamic_hyperparameters,
            ),
            TensorboardLoggingCallback(
                context=context,
                static_hyperparameters=static_hyperparameters,
                dynamic_hyperparameters=dynamic_hyperparameters,
                weights=True,
            ),
        ]

    def _get_learning_rate(self) -> Union[Tuple[str, List[Union[str, int]]], None]:
        """
        Try and get the learning rate value form the stored optimizer.

        :return: The key chain to get the optimizer learning rate value or None if the learning rate could not be
                 accessed via the common key.
        """
        if "lr" in self._optimizer.param_groups[0]:
            return HyperparametersKeys.OPTIMIZER, ["param_groups", 0, "lr"]
        if "learning_rate" in self._optimizer.param_groups[0]:
            return HyperparametersKeys.OPTIMIZER, ["param_groups", 0, "learning_rate"]
        return None

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

    def _train(self, callbacks_handler: CallbacksHandler, use_cuda: bool):
        """
        Initiate a single epoch training.

        :param callbacks_handler: Callbacks handler to use.
        :param use_cuda:          Whether or not to use cuda if available.
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
            if use_cuda and torch.cuda.is_available():
                x = x.cuda()
                y_true = y_true.cuda()

            # Zero the parameters gradients:
            self._optimizer.zero_grad()

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
            callbacks_handler.on_optimizer_step_end()

            # End of batch callbacks:
            if not callbacks_handler.on_train_batch_end(
                batch=batch, x=x, y_true=y_true, y_pred=y_pred
            ):
                break

    def _validate(
        self,
        callbacks_handler: CallbacksHandler,
        use_cuda: bool,
    ) -> Tuple[MetricValueType, List[MetricValueType]]:
        """
        Initiate a single epoch validation.

        :param callbacks_handler: Callbacks handler to use.
        :param use_cuda:          Whether or not to use cuda if available.

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
                if use_cuda and torch.cuda.is_available():
                    x = x.cuda()
                    y_true = y_true.cuda()

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
        metric_values = [
            (sum(metric) / len(metric)) for metric in metrics
        ]  # TODO: Fix division by 0 when no metrics were given
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
