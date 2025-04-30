import functools
import inspect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_calls(log_args: bool = True):
    """
    Decorator to log calls to functions or classes.

    Parameters:
    log_args (bool): If True, log argument values; if False, log only argument names.
    """

    def decorator(obj):
        if inspect.isfunction(obj):
            # It's a function or method
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                arg_info = inspect.signature(obj).bind(*args, **kwargs)
                arg_info.apply_defaults()
                if log_args:
                    args_str = ", ".join(
                        f"{k}={v!r}" for k, v in arg_info.arguments.items()
                    )
                else:
                    args_str = ", ".join(arg_info.arguments.keys())
                logging.info(
                    f" [CLASS-METHOD-CALL-LOG] Calling {obj.__name__}({args_str})"
                )
                result = obj(*args, **kwargs)
                logging.info(f"{obj.__name__} returned {result!r}")
                return result

            return wrapper
        elif inspect.isclass(obj):
            # It's a class
            original_init = obj.__init__

            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                arg_info = inspect.signature(original_init).bind(self, *args, **kwargs)
                arg_info.apply_defaults()
                if log_args:
                    args_str = ", ".join(
                        f"{k}={v!r}" for k, v in list(arg_info.arguments.items())[1:]
                    )
                else:
                    args_str = ", ".join(list(arg_info.arguments.keys())[1:])
                logging.debug(
                    f"Creating instance of {obj.__name__} with args: {args_str}"
                )
                original_init(self, *args, **kwargs)
                logging.debug(f"Instance of {obj.__name__} created: {self!r}")

            obj.__init__ = new_init
            return obj
        else:
            raise TypeError(
                "log_calls decorator can only be applied to functions or classes."
            )

    return decorator
