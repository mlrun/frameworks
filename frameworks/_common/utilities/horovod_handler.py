import sys
from abc import ABC, abstractmethod


class HorovodHandler(ABC):
    """
    Static class interface for handling the global horovod module.
    """

    # The shared module object holding the 'import horovod.{framework} as hvd' command:
    hvd = None

    @staticmethod
    @abstractmethod
    def import_horovod():
        """
        Import horovod globally to the system so one instance of horovod will be shared across all objects.
        """
        pass

    @staticmethod
    def get_horovod():
        """
        Get the global instance of horovod.
        :return: The 'hvd' global instance of horovod. If it was not imported, None will be returned.
        """
        return HorovodHandler.hvd

    @staticmethod
    def delete_horovod():
        """
        Delete the global horovod module from the system. If it was not imported, the function will simply return.
        """
        if not HorovodHandler.hvd:
            return
        del HorovodHandler.hvd
        HorovodHandler.hvd = None
        for module_name in sys.modules:
            if "horovod" in module_name:
                del sys.modules[module_name]

    @staticmethod
    def _import_horovod_framework(framework: str):
        """
        Import the global instance of horovod according to the given framework.
        :param framework: The framework to import. Can be one of 'torch', 'tensorflow.keras' and 'tensorflow'.
        """
        HorovodHandler.hvd = __import__("horovod.{}".format(framework))
