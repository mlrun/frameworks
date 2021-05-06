from abc import ABC, abstractmethod


class ModelHandler(ABC):
    """
    An abstract interface for handling a model of the supported frameworks.
    """

    def __init__(self, model=None):
        """
        Initialize the handler. The model can be set here so it won't require loading.
        :param model: Model to handle or None in case a loading parameters were supplied.
        """
        self._model = model

    @property
    def model(self):
        """
        Get the handled model. Will return None in case the model is not initialized.
        :return: The handled model.
        """
        return self._model

    @abstractmethod
    def save(self, output_path: str, *args, **kwargs):
        """
        Save the handled model at the given output path.
        :param output_path: The full path to the directory and the model's file in it to save the handled model at.
        :raise RuntimeError: In case there is no model initialized in this handler.
        """
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        """
        Load the specified model in this handler.
        """
        pass
