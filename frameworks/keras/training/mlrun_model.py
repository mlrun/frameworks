from typing import Union, List, Dict, Tuple, Callable
from types import MethodType, FunctionType
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    Callback,
    ModelCheckpoint,
    TensorBoard,
    ProgbarLogger,
    CSVLogger,
    BaseLogger,
)
from tensorflow.keras.optimizers import Optimizer

import mlrun
from mlrun import MLClientCtx
from frameworks.keras.callbacks import (
    MLRunLoggingCallback,
    TensorboardLoggingCallback,
    TrackableType,
)


class MLRunModel(keras.Model):
    """
    MLRun model is for enabling additional features supported by MLRun in keras. With MLRun model one can apply horovod
    and use auto logging with ease.
    """

    @staticmethod
    def wrap(model: keras.Model) -> keras.Model:
        """
        Wrap the given model with MLRun model features - including its attributes and methods.

        :param model: The model to wrap.

        :return: The wrapped model.
        """
        # Add the MLRun model properties:
        setattr(model, "_callbacks", [])
        setattr(model, "_hvd", None)

        # Add the MLRun model methods:
        for method_name in ["auto_log", "use_horovod", "note_rank_0_callback"]:
            if method_name not in model.__dir__():
                setattr(model, method_name, MethodType(getattr(MLRunModel, method_name), model))

        # Wrap the compile method:
        def compile_wrapper(method):
            def wrapper(*args, **kwargs):
                # Call the pre compile method:
                (
                    kwargs["optimizer"],
                    experimental_run_tf_function,
                ) = MLRunModel._pre_compile(self=model, optimizer=kwargs["optimizer"])
                # Assign parameters:
                if experimental_run_tf_function is not None:
                    kwargs["experimental_run_tf_function"] = experimental_run_tf_function
                # Call the original compile method:
                method(*args, **kwargs)

            return wrapper

        setattr(model, "compile", compile_wrapper(model.compile))

        # Wrap the fit method:
        def fit_wrapper(method):
            def wrapper(*args, **kwargs):
                # Setup the callbacks list:
                if "callbacks" not in kwargs or kwargs["callbacks"] is None:
                    kwargs["callbacks"] = []
                # Add auto log callbacks if they were added:
                kwargs["callbacks"] = kwargs["callbacks"] + model._callbacks
                # Setup default values if needed:
                if "verbose" not in kwargs:
                    kwargs["verbose"] = 1
                if "steps_per_epoch" not in kwargs:
                    kwargs["steps_per_epoch"] = None
                if "validation_steps" not in kwargs:
                    kwargs["validation_steps"] = None
                # Call the pre fit method:
                (
                    kwargs["callbacks"],
                    kwargs["verbose"],
                    kwargs["steps_per_epoch"],
                    kwargs["validation_steps"],
                ) = MLRunModel._pre_fit(
                    self=model,
                    optimizer=model.optimizer,
                    callbacks=kwargs["callbacks"],
                    verbose=kwargs["verbose"],
                    steps_per_epoch=kwargs["steps_per_epoch"],
                    validation_steps=kwargs["validation_steps"],
                )
                method(*args, **kwargs)

            return wrapper

        setattr(model, "fit", fit_wrapper(model.fit))

        return model

    def __init__(self, model_to_wrap: keras.Model = None, *args, **kwargs):
        """
        Initialize a MLRun model with the additional features of MLRun in keras.
        """
        super(MLRunModel, self).__init__(*args, **kwargs)

        # Setup MLRun model attributes:
        self._wrapped_model = model_to_wrap
        self._callbacks = []
        self._hvd = None

    def auto_log(
        self,
        context: MLClientCtx,
        static_hyperparameters: Dict[
            str, Union[TrackableType, List[Union[str, int]]]
        ] = None,
    ):
        """
        Initialize the defaulted logging callbacks by MLRun. Given the context, the method will setup a list of
        callbacks with the most common settings for logging a training session in tensorflow.keras. For further
        information regarding the logging callbacks, see 'mlrun.frameworks.keras.callbacks.MLRunLoggingCallback' and
        'mlrun.frameworks.keras.callbacks.TensorboardLoggingCallback'.

        :param context:                The MLRun context to log with.
        :param static_hyperparameters: A dictionary of static hyperparameters to note in the logs. The parameter expects
                                       a dictionary where the keys are the hyperparameter chosen names and the values
                                       are the hyperparameter static value or a key chain - a list of keys and indices
                                       to know how to access the needed hyperparameter. For example, to track the
                                       'epsilon' attribute of an optimizer and the 'epochs' of an experiment run, one
                                       should pass:
                                       {
                                           "epsilon": ["optimizer", "epsilon"],
                                           "epochs": 7
                                       }
        """
        dynamic_hyperparameters = {"learning_rate": ["optimizer", "lr"]}
        self._callbacks.append(
            MLRunLoggingCallback(
                context=context,
                static_hyperparameters=static_hyperparameters,
                dynamic_hyperparameters=dynamic_hyperparameters,
            )
        )
        self._callbacks.append(
            TensorboardLoggingCallback(
                context=context,
                static_hyperparameters=static_hyperparameters,
                dynamic_hyperparameters=dynamic_hyperparameters,
                weights=True,
            )
        )

    def use_horovod(self):
        """
        Setup the model or wrapped model to run with horovod.
        """
        # Import horovod:
        import horovod.tensorflow.keras as hvd

        # Initialize horovod:
        hvd.init()

        # Link the horovod to the class pointer:
        self._hvd = hvd

    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs
    ):
        """
        Compile the model or, if given, the wrapped model with the given parameters (see
        tensorflow.keras.Model.compile() for better reference of the compile method). Notice, for compiling the model to
        run with horovod the optimizer given must be an initialized instance of 'tensorflow.keras.optimizers.Optimizer'
        and not a string.

        :param optimizer:           An optimizer instance. Note that when using horovod, unlike in
                                    'tf.keras.Model.compile()', the optimizer must be an initialized instance of an
                                    optimizer and it cannot be passed as a string.
        :param loss:                Loss function as in 'tensorflow.keras.Model.compile()'.
        :param metrics:             List of metric functions as in 'tf.keras.Model.compile()'.
        :param loss_weights:        A list of scalars to apply on the outputs or a dictionary of output names to scalars
                                    as in 'tf.keras.Model.compile()'.
        :param weighted_metrics:    List of metric functions as in 'tf.keras.Model.compile()'.
        :param run_eagerly:         Whether or not to wrap the model in 'tf.function' as in 'tf.keras.Model.compile()'.
        :param steps_per_execution: Number of steps to run on each 'tf.function' call as in 'tf.keras.Model.compile()'.
        """
        # Call the pre compile method:
        optimizer, experimental_run_tf_function = self._pre_compile(optimizer=optimizer)
        if experimental_run_tf_function is not None:
            kwargs["experimental_run_tf_function"] = experimental_run_tf_function

        # Call the compile method of the super class:
        super(MLRunModel, self).compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            **kwargs
        )

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        """
        Trains the model or wrapped model if given (see tensorflow.keras.Model.fit() for better reference of the fit
        method).

        :param x:                     The input data.
        :param y:                     The target data.
        :param batch_size:            The batch size.
        :param epochs:                Amount of epochs for training.
        :param verbose:               Output for training process.
        :param callbacks:             List of callbacks to apply during training.
        :param validation_split:      Float number that indicates to use a fraction of the training data as validation
                                      data.
        :param validation_data:       The validation data.
        :param shuffle:               Boolean for whether to shuffle the training data or not.
        :param class_weight:          Dictionary for mapping class indices to a weight value for weighting the loss
                                      function during training.
        :param sample_weight:         Numpy array of weights used for weighting the loss function during training.
        :param initial_epoch:         The epoch to start training.
        :param steps_per_epoch:       Amount of training steps to perform each epoch.
        :param validation_steps:      Amount of validation steps to perform each epoch.
        :param validation_batch_size: The validation data batch size.
        :param validation_freq:       Per how many peochs to run validation. Can be sent as a container.
        :param max_queue_size:        The maximum size for the generator queue.
        :param workers:               The maximum number of workers to use when using process-based threading.
        :param use_multiprocessing:   Whether or not to use process-based threading.

        :return: History object.
        """
        # Setup the callbacks list:
        if callbacks is None:
            callbacks = []

        # Add auto log callbacks if they were added:
        callbacks = callbacks + self._callbacks

        # Call the pre fit method:
        (callbacks, verbose, steps_per_epoch, validation_steps,) = self._pre_fit(
            optimizer=(
                self.optimizer
                if self._wrapped_model is None
                else self._wrapped_model.optimizer
            ),
            callbacks=callbacks,
            verbose=verbose,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )

        # Call the fit method of the super class:
        super(MLRunModel, self).fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks + self._callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_batch_size=validation_batch_size,
            validation_freq=validation_freq,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )

    def call(self, inputs, training=None, mask=None):
        """
        Performs the logic of applying the model or, if given, the wrapped model to the input tensors. For further
        information, see the docs of keras.Model.

        :param inputs:   The input tensors.
        :param training: Whether the call is in inference mode or training mode.
        :param mask:     Boolean tensor encoding masked timesteps in the input, used in RNN layers.
        """
        super(MLRunModel, self).call(inputs=inputs, training=training, mask=mask)

    def get_config(self) -> dict:
        """
        Returns a dictionary containing the configuration used to initialize this model or, if given, the wrapped model.
        For further information, see the docs of keras.Model.

        :return: The configuration dictionary of this model or the wrapped model if provided.
        """
        return super(MLRunModel, self).get_config()

    def note_rank_0_callback(self, callback_name: str):
        """
        Note an additional custom callback to be applied only on rank 0 when using horovod.

        :param callback_name: The name of the callback.
        """
        self._RANK_0_ONLY_CALLBACKS.append(callback_name)

    # List of all the methods that should not be overridden when wrapping a model:
    _MLRUN_MODEL_METHODS = [
        auto_log.__name__,
        compile.__name__,
        fit.__name__,
        call.__name__,
        get_config.__name__,
    ]  # type: List[str]

    # List of all the callbacks that should only be applied on rank 0 when using horovod:
    _RANK_0_ONLY_CALLBACKS = [
        MLRunLoggingCallback.__name__,
        TensorboardLoggingCallback.__name__,
        ModelCheckpoint.__name__,
        TensorBoard.__name__,
        ProgbarLogger.__name__,
        CSVLogger.__name__,
        BaseLogger.__name__,
        "__class__",
    ]  # type: List[str]

    def _pre_compile(self, optimizer: Optimizer) -> Tuple[Optimizer, Union[bool, None]]:
        """
        Method to call before calling 'compile' to setup the run and inputs for using horovod.

        :param optimizer: The optimzier to compile. It will be wrapped in horovod's distributed optimizer:
                          'hvd.DistributedOptimizer'.

        :return: The updated parameters:
                 [0] = Wrapped optimizer.
                 [1] = The 'experimental_run_tf_function' parameter for 'compile' kwargs or 'None' if horovod should not
                       be used.

        :raise ValueError: In case the optimizer was passed as a string.
        """
        # Check if needed to run with horovod:
        if self._hvd is None:
            return optimizer, None

        # Validate the optimizer input:
        if isinstance(optimizer, str):
            raise ValueError(
                "When using horovod, the compile mehotd is expecting an initialized optimizer "
                "instance and not a string."
            )

        # Setup the device to run on GPU if available:
        if tf.config.experimental.list_physical_devices("GPU"):
            # Pin each GPU to a single process:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(
                    gpus[self._hvd.local_rank()], "GPU"
                )
        else:
            # No GPUs were found, or 'use_cuda' was false:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # Adjust learning rate based on the number of GPUs:
        optimizer.lr = optimizer.lr * self._hvd.size()

        # Wrap the optimizer in horovod's distributed optimizer: 'hvd.DistributedOptimizer'.
        optimizer = self._hvd.DistributedOptimizer(optimizer)

        # Compile the model with `experimental_run_tf_function=False` to ensure Tensorflow uses the distributed
        # optimizer to compute gradients:
        experimental_run_tf_function = False

        return optimizer, experimental_run_tf_function

    def _pre_fit(
        self,
        optimizer: Optimizer,
        callbacks: List[Callback],
        verbose: int,
        steps_per_epoch: Union[int, None],
        validation_steps: Union[int, None],
    ) -> Tuple[List[Callback], int, Union[int, None], Union[int, None]]:
        """
        Method to call before calling 'fit' to setup the run and inputs for using horovod.

        :param optimizer:        Optimizer to get his initial learning rate for one of horovod's callbacks.
        :param callbacks:        Callbacks to use in the run. The callbacks will be split among the ranks so only
                                 certain callbacks (mainly logging and checkpoints) will be in rank 0.
        :param verbose:          Whether or not to print the progress of training. If '1' or '2' only rank 0 will be
                                 applied with the verbose.
        :param steps_per_epoch:  Amount of training steps to run in each epoch. The steps will be divided by the size of
                                 ranks (horovod workers).
        :param validation_steps: Amount of validation steps to run in each epoch. The steps will be divided by the size
                                 of ranks (horovod workers).

        :return: The updated parameters according to the used rank:
                 [0] = Callbacks list.
                 [1] = Verbose
                 [2] = Steps per epoch or None if not given.
                 [3] = Validation steps or None if not given.
        """
        # Check if needed to run with horovod:
        if self._hvd is None:
            return callbacks, verbose, steps_per_epoch, validation_steps

        # Setup the callbacks:
        metric_average_callback = self._hvd.callbacks.MetricAverageCallback()
        metric_average_callback._supports_tf_logs = True
        horovod_callbacks = [
            self._hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            metric_average_callback,
            self._hvd.callbacks.LearningRateWarmupCallback(
                initial_lr=float(optimizer.lr)
            ),
        ]
        if self._hvd.rank() != 0:
            callbacks = [
                callback
                for callback in callbacks
                if type(callback).__name__ not in self._RANK_0_ONLY_CALLBACKS
            ]
        callbacks = horovod_callbacks + callbacks

        # Pick the verbose:
        if self._hvd.rank() != 0:
            verbose = 0

        # Adjust the number of steps per epoch based on the number of GPUs (if given):
        if steps_per_epoch is not None:
            steps_per_epoch = steps_per_epoch // self._hvd.size()
        if validation_steps is not None:
            validation_steps = validation_steps // self._hvd.size()

        return callbacks, verbose, steps_per_epoch, validation_steps
