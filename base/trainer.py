from abc import ABC, abstractmethod


class Trainer(ABC):
    """
    An interface for a trainer - a class for wrapping a framework training process supplying additional mlrun
    features.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the trainer. All training main objects such as the model, datasets, loss, metrics and optimizers
        should be passed here.
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run mlrun's trainer for the specific framework, training a model with additional convenient features. Callbacks,
        epochs and iterations should be passed here.
        """
        pass

    @abstractmethod
    def auto_log(self, *args, **kwargs):
        """
        Run training with automatic logging to mlrun's context and tensorboard.
        """
        pass
