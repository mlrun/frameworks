from typing import List, Dict, Any, Type
from types import MethodType
from abc import ABC, abstractmethod


class MLRunInterface(ABC):

    # Properties attributes to be inserted so the keras mlrun interface will be fully enabled:
    _PROPERTIES = {}  # type: Dict[str, Any]

    # Methods attributes to be inserted so the keras mlrun interface will be fully enabled:
    _METHODS = []  # type: List[str]

    # Functions attributes to be inserted so the keras mlrun interface will be fully enabled:
    _FUNCTIONS = []  # type: List[str]

    @classmethod
    @abstractmethod
    def add_interface(cls, model: object, *args, **kwargs):
        # Add the MLRun properties:
        cls._insert_properties(model=model, interface=cls)

        # Add the MLRun methods:
        cls._insert_methods(model=model, interface=cls)

        # Add the MLRun functions:
        cls._insert_functions(model=model, interface=cls)

    @staticmethod
    def _insert_properties(model, interface: Type['MLRunInterface']):
        for property_name, default_value in interface._PROPERTIES.items():
            if property_name not in model.__dir__():
                setattr(model, property_name, default_value)

    @staticmethod
    def _insert_methods(model, interface: Type['MLRunInterface']):
        for method_name in interface._METHODS:
            if method_name not in model.__dir__():
                setattr(
                    model,
                    method_name,
                    MethodType(getattr(interface, method_name), model),
                )

    @staticmethod
    def _insert_functions(model, interface: Type['MLRunInterface']):
        for function_name in interface._FUNCTIONS:
            if function_name not in model.__dir__():
                setattr(
                    model,
                    function_name,
                    getattr(interface, function_name),
                )
