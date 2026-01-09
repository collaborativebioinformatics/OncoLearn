import hashlib

import numpy as np

_B32 = 32


class ListSerializer:
    """
    Serializer for list of elements.

    Args:
    ----
        element_serializer: Serializer for the elements.
    """

    int_size_bytes = 4

    def __init__(self, element_serializer) -> None:
        self.element_serializer = element_serializer

    def serialize(self, obj: list) -> bytes:
        """

        Serialize a list of elements.

        Args:
        ----
            obj (list): List of elements to serialize.

        Returns:
        -------
                bytes: Serialized data.

        """
        n = len(obj)
        data = [np.uint32(n).tobytes()]
        for i in obj:
            element = self.element_serializer.serialize(i)
            element_size = len(element)
            data.append(np.uint32(element_size).tobytes())
            data.append(element)

        return b"".join(data)

    def deserialize(self, data: bytes) -> list:
        """
        Deserialize a list of elements.

        Args:
        ----
            data (bytes): Serialized data.

        Returns:
        -------
            list: Deserialized list of elements.
        """
        n = np.frombuffer(data[: ListSerializer.int_size_bytes], np.uint32).item()
        offset = ListSerializer.int_size_bytes
        result = []
        for i in range(n):
            element_size = np.frombuffer(data[offset : offset + 4], np.uint32).item()
            offset += ListSerializer.int_size_bytes
            element = self.element_serializer.deserialize(
                data[offset : offset + element_size]
            )
            result.append(element)
            offset += element_size

        return result


class StrElementSerializer:
    """Serializer for string elements."""

    def serialize(self, obj: str) -> bytes:
        """
        Serialize a string element.

        Args:
        ----
            obj (str): String element to serialize.
        """
        return obj.encode()

    def deserialize(self, data: bytes) -> list:
        """
        Deserialize a string element.

        Args:
        ----
            data (bytes): Serialized data.
        """
        return data.decode()


class ListStrElementSerializer(ListSerializer):
    """Serializer for list of string elements."""

    def __init__(self):
        """Initialize the serializer."""
        str_serializer = StrElementSerializer()
        super().__init__(str_serializer)


class HashSerializer:
    @staticmethod
    def serialize(data: bytes):
        hash = hashlib.sha256(data).digest()
        return hash + data

    @staticmethod
    def deserialize(data: bytes):
        data_hash = data[:_B32]
        data = data[_B32:]
        if hashlib.sha256(data).digest() != data_hash:
            raise OSError("Hash is wrong, data is corrupted.")
        return data


class IndexSerializer:
    version = b"0.0"

    @staticmethod
    def serialize(index: list[int]):
        index = np.array(index, dtype=np.int64)
        data = index.tobytes()
        data = HashSerializer.serialize(data)
        return IndexSerializer.version + data

    @staticmethod
    def deserialize(data: bytes):
        data = data[len(IndexSerializer.version) :]
        data = HashSerializer.deserialize(data)
        data = np.frombuffer(data, dtype=np.int64)
        return data
