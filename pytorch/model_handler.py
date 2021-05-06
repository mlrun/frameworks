from typing import Type
import importlib.util
import torch
from torch.nn import Module
from base.model_handler import ModelHandler


class PyTorchModelHandler(ModelHandler):
    def __init__(
        self,
        model: Module = None,
        model_class: Type[Module] = None,
        model_class_name: str = None,
        model_py_file_path: str = None,
        model_pt_file_path: str = None,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading.
        :param model:              Model to handle or None in case a loading parameters were supplied.
        :param model_class:        The model's class if it is already imported (known during the run).
        :param model_class_name:   The class name of the model's class if it is not imported (unknown and imported from
                                   file).
        :param model_py_file_path: The path to the python file having the class code if the model's class is not
                                   imported (unknown and imported from file).
        :param model_pt_file_path: The model's saved '.pt' file with its tensors and attributes to load.
        """
        # Set the model if given:
        super(PyTorchModelHandler, self).__init__(model=model)

        # Store the model properties:
        self._model = model
        self._class = model_class
        self._pt_file_path = model_pt_file_path

        # If the model's class name and py file were given, import them so the class will be ready to use for loading:
        if model_class_name and model_py_file_path:
            self._import_module(
                class_name=model_class_name, py_file_path=model_py_file_path
            )

    def save(self, output_path: str, *args, **kwargs):
        """
        Save the handled model at the given output path.
        :param output_path: The full path to the directory and the model's file in it to save the handled model at.
        :raise RuntimeError: In case there is no model initialized in this handler.
        """
        if self._model is None:
            raise RuntimeError(
                "Model cannot be save as it was not given in initialization or loaded during this run."
            )
        torch.save(self._model.state_dict(), output_path)

    def load(self, *args, **kwargs):
        """
        Load the specified model in this handler. Additional parameters for the class initializer can be passed via the
        args list and kwargs dictionary.
        """
        # If a model instance is already loaded, delete it from memory:
        if self._model:
            del self._model

        # Initialize the model:
        self._model = self._class(*args, **kwargs)

        # Load the state dictionary into it:
        self._model.load_state_dict(torch.load(self._pt_file_path))

    def _import_module(self, class_name: str, py_file_path: str):
        """
        Import the model's class from the given python file as: from PY_FILE_PATH import CLASS_NAME
        :param class_name:   The class name to be imported from the given python file.
        :param py_file_path: Path to the python file with the class code.
        """
        # Import the class:
        spec = importlib.util.spec_from_file_location(
            name=class_name, location=py_file_path
        )
        module = importlib.util.module_from_spec(spec=spec)
        spec.loader.exec_module(module)

        # Store the imported class:
        self._class = getattr(module, class_name)
