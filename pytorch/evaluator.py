from base.evaluator import Evaluator


class PyTorchEvaluator(Evaluator):
    """
    An interface for an evaluator - a class for wrapping a pytorch model evaluation process supplying additional mlrun
    features.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the evaluator. All evaluation main objects such as the model, dataset, loss and metrics should be
        passed here.
        """
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """
        Run mlrun's pytorch evaluator on the initialized objects. Callbacks and iterations should be passed here.
        """
        raise NotImplementedError

    def auto_log(self, *args, **kwargs):
        """
        Run an evaluation with automatic logging to mlrun's context and tensorboard.
        """
        raise NotImplementedError
