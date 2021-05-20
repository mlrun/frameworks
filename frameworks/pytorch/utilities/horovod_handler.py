from frameworks._common.utilities import HorovodHandler


class PyTorchHorovodHandler(HorovodHandler):
    """
    Static class for handling the horovod.torch module.
    """

    @staticmethod
    def import_horovod():
        """
        Import horovod.torch globally to the system so one instance of horovod will be shared across all objects. If the
        module was already imported, the function will simply return.
        """
        if PyTorchHorovodHandler.hvd:
            return
        PyTorchHorovodHandler._import_horovod_framework(framework='torch')
