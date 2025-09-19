from typing import Tuple, Union, Dict, Type, Any


def check_type(params_and_types: Dict[str, Tuple[Any, Union[Type, Tuple[Type, ...]], bool]]):
    """
    Validates that each parameter value matches its expected type, and handles optional parameters.

    :param params_and_types: A dictionary where each key is a parameter name, and the value is a tuple:
        (value, expected_type(s), is_optional), where:
            - value: The actual value to validate.
            - expected_type(s): A type or tuple of types that the value is expected to match.
            - is_optional: If True, allows the value to be None.

        Example:
            check_type({
                "name": ("Alice", str, False),
                "age": (None, int, True),
                "scores": ([90, 95], list, False),
            })

    :raises TypeError:
        - If a value is None and is_optional is False.
        - If a value does not match the expected type(s).
    """
    for param_name, (value, expected_types, is_optional) in params_and_types.items():
        if value is None:
            if is_optional:
                continue
            else:
                raise TypeError(f"{param_name} must not be None.")

        if not isinstance(value, expected_types):
            allowed_types = (
                expected_types.__name__
                if isinstance(expected_types, type)
                else ', '.join(t.__name__ for t in expected_types)
            )
            raise TypeError(f"{param_name} must be of type {allowed_types}. Got {type(value).__name__}.")

