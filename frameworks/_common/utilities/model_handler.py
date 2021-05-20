from typing import Union, List, Dict, Type
from abc import ABC, abstractmethod
import importlib.util
import mlrun


class ModelHandler(ABC):
    """
    An abstract interface for handling a model of the supported frameworks.
    """

    def __init__(self, model=None, context: mlrun.MLClientCtx = None):
        """
        Initialize the handler. The model can be set here so it won't require loading.
        :param model:   Model to handle or None in case a loading parameters were supplied.
        :param context: MLRun context to work with for automatic loading and saving to the project directory.
        """
        self._model = model
        self._context = context

    @property
    def model(self):
        """
        Get the handled model. Will return None in case the model is not initialized.
        :return: The handled model.
        """
        return self._model

    @abstractmethod
    def save(self, output_path: str = None, *args, **kwargs):
        """
        Save the handled model at the given output path.
        :param output_path:  The full path to the directory / model's file to save the handled model at. If not given
                             The context stored will be used to save the model in the defaulted location.
        :raise RuntimeError: In case there is no model initialized in this handler.
        :raise ValueError:   If an output path was not given, yet a context was not provided in initialization.
        """
        if self._model is None:
            raise RuntimeError(
                "Model cannot be save as it was not given in initialization or loaded during this run."
            )
        if output_path is None and self._context is None:
            raise ValueError(
                "An output path was not given and a context was not provided during the initialization of "
                "this model handler. To save the model, one of the two parameters must be supplied."
            )

    @abstractmethod
    def load(self, uid: str = None, epoch: int = None, *args, **kwargs):
        """
        Load the specified model in this handler. If a context was provided during initialization, the defaulted version
        of the model in the project will be loaded. To specify the model's version, its uid can be supplied along side
        an epoch for loading a callback of this run.
        :param uid:   To load a specific version of the model by the run uid that generated the model.
        :param epoch: To load a checkpoint of a given training (training's uid), add the checkpoint's epoch number.
        :raise ValueError: If a context was not provided during the handler initialization yet a uid was provided or if
                           an epoch was provided but a uid was not.
        """
        # Validate input:
        # # Epoch [V], UID [X]:
        if epoch is not None and uid is None:
            raise ValueError(
                "To load a model from a checkpoint of an epoch, the training run uid must be given."
            )
        # # Epoch [?], UID [V], Context [X]:
        if uid is not None and self._context is None:
            raise ValueError(
                "To load a specific version (by uid) of a model a context must be provided during the "
                "handler initialization."
            )

    @abstractmethod
    def log(self):
        if self._context is None:
            raise ValueError(
                "Cannot log model if a context was not provided during initialization."
            )
        # TODO: Implement the log model here!

    def _get_model_directory(self, uid: Union[str, None], epoch: Union[int, None]):
        # TODO: Need to decide on the file system architecture. Where will the project's models versions be saved. The
        #       current idea is for the models to be specified by their function's uid or a version string (more
        #       suitable...) and for each project there will be a models.json file where each model will have his own
        #       default version to be loaded.
        pass

    @staticmethod
    def _import_module(classes_names: List[str], py_file_path: str) -> Dict[str, Type]:
        """
        Import the given class by its name from the given python file as: from 'py_file_path' import 'class_name'.
        :param classes_names: The classes names to be imported from the given python file.
        :param py_file_path:  Path to the python file with the classes code.
        :return: The imported classes dictionary where the keys are the classes names and the values are their imported
                 classes.
        """
        # Initialize the imports dictionary:
        classes_imports = {}

        # Go through the given classes:
        for class_name in classes_names:
            # Import the class:
            spec = importlib.util.spec_from_file_location(
                name=class_name, location=py_file_path
            )
            module = importlib.util.module_from_spec(spec=spec)
            spec.loader.exec_module(module)
            # Get the imported class and store it:
            classes_imports[class_name] = getattr(module, class_name)

        return classes_imports
