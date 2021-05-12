from typing import Type
from abc import ABC, abstractmethod
import importlib.util

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

    def _import_module(self, class_name: str, py_file_path: str) -> Type:
        """
        Import the given class by its name from the given python file as: from 'py_file_path' import 'class_name'.
        :param class_name:   The class name to be imported from the given python file.
        :param py_file_path: Path to the python file with the class code.
        :return: The imported class.
        """
        # Import the class:
        spec = importlib.util.spec_from_file_location(
            name=class_name, location=py_file_path
        )
        module = importlib.util.module_from_spec(spec=spec)
        spec.loader.exec_module(module)

        # Get the imported class and return it:
        return getattr(module, class_name)
