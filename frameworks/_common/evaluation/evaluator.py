from abc import ABC, abstractmethod


class Evaluator(ABC):
    """
    An interface for an evaluator - a class for wrapping a framework evaluation process supplying additional mlrun
    features.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the evaluator. All evaluation main objects such as the model, dataset, loss and metrics should be
        passed here.
        """

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run mlrun's evaluator for the specific framework, evaluating a model with additional convenient features.
        Callbacks and iterations should be passed here.
        """
        pass

    @abstractmethod
    def auto_log(self, *args, **kwargs):
        """
        Run an evaluation with automatic logging to mlrun's context and tensorboard.
        """
        pass
