from typing import Union, List, Callable

from frameworks.pytorch.utilities.horovod_handler import HorovodHandler
from frameworks.pytorch.utilities.model_handler import PyTorchModelHandler
from frameworks.pytorch.utilities.callbacks_handler import CallbacksHandler


def with_horovod(worker: Union[List[int], int] = None):
    """
    A decorator for running a function from a specific worker using horovod. If horovod is not found / imported in the
    interpreter the function will run by default.
    :param worker: The worker that should run the function. Can be passed as a list for multiple workers.
    :return: The decorated function's result if it ran and nothing otherwise.
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Check if a specific worker is given, if not, run on all workers:
            if worker is None:
                return func(*args, **kwargs)
            # Check if horovod is imported, if not, run by default:
            hvd = HorovodHandler.get_horovod()
            if hvd is None:
                return func(*args, **kwargs)
            # Check if the current worker matches the specified worker. If so, run the function:
            if isinstance(worker, int) and hvd.local_rank() == worker:
                return func(*args, **kwargs)
            elif hvd.local_rank() in worker:
                return func(*args, **kwargs)
            # Do not run the decorated function:
            return

        return wrapper

    return decorator
