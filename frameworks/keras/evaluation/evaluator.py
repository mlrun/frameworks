from frameworks._common.evaluation.evaluator import Evaluator


class KerasEvaluator(Evaluator):
    """
    An interface for an evaluator - a class for wrapping a tensorflow.keras model evaluation process supplying
    additional mlrun features.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the evaluator. All evaluation main objects such as the model, dataset, loss and metrics should be
        passed here.
        """
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """
        Run mlrun's keras evaluator on the initialized objects. Callbacks should be passed here.
        """
        raise NotImplementedError

    def auto_log(self, *args, **kwargs):
        """
        Run an evaluation with automatic logging to mlrun's context and tensorboard.
        """
        raise NotImplementedError