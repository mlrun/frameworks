from typing import Union, Dict, Type
import importlib.util

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.metrics import Metric

from frameworks.base.model_handler import ModelHandler


class SaveFormats:
    """
    Save formats to pass to the 'KerasModelHandler'.
    """

    SAVED_MODEL_DIRECTORY_FORMAT = "SavedModel"
    JSON_ARCHITECTURE_H5_WEIGHTS_FORMAT = "json_h5"


class KerasModelHandler(ModelHandler):
    def __init__(
        self,
        model: Model = None,
        model_file_path: str = None,
        weights_file_path: str = None,
        custom_objects: Dict[
            str,
            Union[
                str, Type[Model], Type[Layer], Type[Loss], Type[Optimizer], Type[Metric]
            ],
        ] = None,
        save_format: str = SaveFormats.JSON_ARCHITECTURE_H5_WEIGHTS_FORMAT,
        save_traces: bool = False,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading.
        :param model:             Model to handle or None in case a loading parameters were supplied.
        :param model_path:   Path to the
        :param weights_path: T
        :param custom_objects:    A dictionary of all the custom objects required for loading the model. The keys are
                                  the class name of the custom object and the value can be the class or a path to a
                                  python file for the handler to import the class from. Notice, if the model was saved
                                  with the 'save_traces' flag on (True) the custom objects are not needed for loading
                                  the model, but each of the custom object must implement the methods 'get_config' and
                                  'from_config'.
        :param save_format:       The save format to use. Should be passed as a member of the class 'SaveFormats'.
        :param save_traces:       Whether or not to use functions saving (only available for the save format
                                  'SaveFormats.SAVED_MODEL_DIRECTORY_FORMAT') for loading the model later without the
                                  custom objects dictionary.
        """
        # Set the model if given:
        super(KerasModelHandler, self).__init__(model=model)

        raise NotImplementedError

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
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """
        Load the specified model in this handler. Additional parameters for the class initializer can be passed via the
        args list and kwargs dictionary.
        """
        # If a model instance is already loaded, delete it from memory:
        if self._model:
            del self._model

        raise NotImplementedError
