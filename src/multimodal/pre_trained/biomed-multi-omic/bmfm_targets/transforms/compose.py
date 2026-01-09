from collections.abc import Callable
from typing import Any


class Compose:
    """
    Compose class represents a composition of multiple transformations.

    It takes a list of callable transformations and applies them
    sequentially.
    """

    def __init__(self, transforms: list[Callable]):
        """
        Initializes the Compose object with a list of transformations.

        Args:
        ----
            transforms (List[Callable]): List of callable transformations.
        """
        super().__init__()
        self.transforms = transforms

    def __call__(self, *args, **kwargs) -> Any:
        """
        Applies the composed transformations to the input data.

        Args:
        ----
            *args: Positional arguments (not used).
            **kwargs: Keyword arguments containing the input data.

        Returns:
        -------
            any: Transformed data.

        Raises:
        ------
            KeyError: If data is not passed as named parameters.
        """
        if args:
            raise KeyError("Please pass data as named parameters.")

        for t in self.transforms:
            kwargs = t(**kwargs)
        return kwargs

    def __repr__(self) -> str:
        """
        Returns a string representation of the Compose object.

        Returns
        -------
            str: String representation of the Compose object.
        """
        format_string = self.__class__.__name__ + "("
        for i, t in enumerate(self.transforms):
            if i:
                format_string += ","
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
