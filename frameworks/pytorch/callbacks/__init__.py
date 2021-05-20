from typing import List
import mlrun

from frameworks.pytorch.callbacks.callback import (
    Callback,
    MetricFunctionType,
    MetricValueType,
)
from frameworks.pytorch.callbacks.logging_callback import (
    LoggingCallback,
    HyperparametersKeys,
)
from frameworks.pytorch.callbacks.mlrun_logging_callback import MLRunLoggingCallback
from frameworks.pytorch.callbacks.tensorboard_logging_callback import (
    TensorboardLoggingCallback,
)


def auto_log_callbacks(context: mlrun.MLClientCtx) -> List[Callback]:
    """
    Get the MLRun's PyTorch framework default logging loggers. Using the context both tensorboard and MLRun logs will
    be provided with their default settings.
    :param context: The MLRun context to use.
    :return: MLRun's PyTorch framework default logging loggers.
    """
    return [
        MLRunLoggingCallback(context=context),
        TensorboardLoggingCallback(context=context),
    ]
