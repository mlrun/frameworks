from typing import Union, List, Dict, Type
import torch
from torch.nn import Module
import mlrun
from frameworks._common.utilities.model_handler import ModelHandler


class PyTorchModelHandler(ModelHandler):
    """
    Class for handling a PyTorch model, enabling loading and saving it during runs.
    """

    def __init__(
        self,
        model_class: Union[Type[Module], str],
        custom_objects: Dict[Union[str, List[str]], str],
        model: Module = None,
        pt_file_path: str = None,
        context: mlrun.MLClientCtx = None
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading.
        :param model_class:    The model's class type object. Can be passed as the class's name (string) as well.
        :param custom_objects: Custom objects the model is using. Expecting a dictionary with the classes names to
                               import as keys (if multiple classes needed to be imported from the same py file a
                               list can be given) and the python file from where to import them as their values. The
                               model class itself must be specified in order to properly save it for later being loaded
                               with a handler. For exmaple:
                               {
                                   "class_name": "/path/to/model.py",
                                   ["layer1", "layer2"]: "/path/to/custom_layers.py"
                               }
        :param model:          Model to handle or None in case a loading parameters were supplied.
        :param pt_file_path:   The model's saved '.pt' file with its tensors and attributes to load.
        :param context:        Context to save, load and log the model.
        """
        # Set the model if given:
        super(PyTorchModelHandler, self).__init__(model=model,
                                                  context=context)

        # Setup the initial model properties:
        self._custom_objects_sources = custom_objects if custom_objects is not None else {}
        self._imported_custom_objects = {}  # type: Dict[str, Type]
        self._pt_file_path = pt_file_path
        self._class_name = None  # type: str
        self._class_py_file = None  # type: str

        # If the model's class name was given, import it so the class will be ready to use for loading:
        if isinstance(model_class, str):
            # Validate input:
            if model_class not in self._custom_objects_sources:
                raise KeyError(
                    "Model class was given by name, yet its py file is not available in the custom objects "
                    "dictionary. The custom objects must have the model's class name as key with the py "
                    "file path as his value."
                )
            # Pop the model's import information:
            self._class_name = model_class
            self._class_py_file = self._custom_objects_sources.pop(model_class)
            # Import the custom objects:
            self._import_custom_objects()
            # Import the model:
            self._class = self._import_module(
                classes_names=[self._class_name], py_file_path=self._class_py_file
            )[self._class_name]
        else:
            # Model is imported, store its class:
            self._class = model_class
            self._class_name = model_class.__name__

    def save(self, output_path: str = None, *args, **kwargs):
        """
        Save the handled model at the given output path.
        :param output_path: The full path to the directory and the model's file in it to save the handled model at.
        :raise RuntimeError: In case there is no model initialized in this handler.
        """
        super(PyTorchModelHandler, self).save(output_path=output_path)
        torch.save({
            self._model.state_dict()
        }, output_path)

    def load(self, uid: str = None, epoch: int = None, *args, **kwargs):
        """
        Load the specified model in this handler. Additional parameters can be passed to the model class constructor via
        the args and kwargs parameters. If a context was provided during initialization, the defaulted version
        of the model in the project will be loaded. To specify the model's version, its uid can be supplied along side
        an epoch for loading a callback of this run.
        :param uid:   To load a specific version of the model by the run uid that generated the model.
        :param epoch: To load a checkpoint of a given training, add the checkpoint's epoch number.
        :raise ValueError: If a context was not provided during the handler initialization yet a uid was provided or if
                           an epoch was provided but a uid was not.
        """
        # If a model instance is already loaded, delete it from memory:
        if self._model:
            del self._model

        # Initialize the model:
        self._model = self._class(*args, **kwargs)

        # Load the state dictionary into it:
        self._model.load_state_dict(torch.load(self._pt_file_path))

    def _import_custom_objects(self):
        if self._custom_objects_sources:
            for classes_names, py_file in self._custom_objects_sources.items():
                self._imported_custom_objects = self._import_module(
                    classes_names=(
                        classes_names
                        if isinstance(classes_names, list)
                        else [classes_names]
                    ),
                    py_file_path=py_file,
                )
