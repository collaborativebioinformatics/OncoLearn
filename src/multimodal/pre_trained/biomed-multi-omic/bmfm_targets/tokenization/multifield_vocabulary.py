import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

from tokenizers import AddedToken


class MultiFieldVocabulary:
    """
    A class for managing a vocabulary of tokens and their corresponding
    IDs.
    """

    def __init__(
        self,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    ):
        """
        Initializes a MultiFieldVocabulary object.

        Args:
        ----
            unk_token (str, optional): The token to use for unknown tokens. Defaults to "[UNK]".
            sep_token (str, optional): The token to use for separating tokens. Defaults to "[SEP]".
            pad_token (str, optional): The token to use for padding. Defaults to "[PAD]".
            cls_token (str, optional): The token to use for the classification token. Defaults to "[CLS]".
            mask_token (str, optional): The token to use for masking. Defaults to "[MASK]".

        Raises:
        ------
            ValueError: If the unk_token is not a string.
            ValueError: If the sep_token is not a string.
            ValueError: If the pad_token is not a string.
            ValueError: If the cls_token is not a string.
            ValueError: If the mask_token is not a string.
        """
        if not isinstance(unk_token, str | AddedToken):
            raise ValueError("unk_token must be a string.")
        if not isinstance(sep_token, str | AddedToken):
            raise ValueError("sep_token must be a string.")
        if not isinstance(pad_token, str | AddedToken):
            raise ValueError("pad_token must be a string.")
        if not isinstance(cls_token, str | AddedToken):
            raise ValueError("cls_token must be a string.")
        if not isinstance(mask_token, str | AddedToken):
            raise ValueError("mask_token must be a string.")

        self.unk_token = str(unk_token)
        self.sep_token = str(sep_token)
        self.pad_token = str(pad_token)
        self.cls_token = str(cls_token)
        self.mask_token = str(mask_token)
        self.global_id: int = 0
        self.token_to_ids: dict[str, dict[str, list[int]]] = OrderedDict()
        self.global_id_to_token: dict[int, list[Any]] = OrderedDict()
        self.local_id_to_token: dict[str, dict[int, str]] = OrderedDict()

    @property
    def special_tokens(self):
        """
        Gets the special tokens.

        Returns
        -------
            list[str]: The special tokens.
        """
        return [
            self.unk_token,
            self.sep_token,
            self.pad_token,
            self.cls_token,
            self.mask_token,
        ]

    def get_special_token_ids(self, field, return_id="local") -> list[int]:
        """
        Gets the IDs for the special tokens.

        Args:
        ----
            field (str): The key for the field.
            return_id (str, optional): Whether to return the local or global ID. Defaults to "local".

        Returns:
        -------
            list[int]: The IDs for the special tokens.
        """
        return [
            self.get_token_id(field, token, return_id) for token in self.special_tokens
        ]

    def add_token_to_vocab(self, field: str, token: str) -> None:
        """
        Adds a token to the vocabulary.

        Args:
        ----
            field (str): The key for the field.
            token (str): The token to add.
        """
        if field not in self.token_to_ids:
            self.token_to_ids[field] = OrderedDict()
        if field not in self.local_id_to_token:
            self.local_id_to_token[field] = OrderedDict()
        global_id = len(self.global_id_to_token)
        local_id = len(self.token_to_ids[field])

        self.token_to_ids[field][token] = [global_id, local_id]
        self.local_id_to_token[field][local_id] = token
        self.global_id_to_token[global_id] = [token, field, local_id]

    def add_tokens_to_vocab(
        self,
        field: str,
        tokens: list[str],
        add_special_tokens: bool = False,
    ) -> None:
        """
        Adds a list of tokens to the vocabulary.

        Args:
        ----
            field (str): The key for the field.
            tokens (list[str]): The tokens to add.
            add_special_tokens (bool, optional): Whether to add the special tokens. Defaults to False.

        Returns:
        -------
            list[int]: The IDs for the tokens.
        """
        if field not in self.token_to_ids:
            self.token_to_ids[field] = OrderedDict()
        if add_special_tokens:
            for token in self.special_tokens:
                self.add_token_to_vocab(field, token)
        for token in tokens:
            self.add_token_to_vocab(field, token)

    def get_token_id(self, field: str, token: str, return_id="local") -> int:
        """
        Gets the ID for a given token and field key.

        Args:
        ----
            field (str): The key for the field.
            token (str): The token to get the ID for.
            return_id (str, optional): Whether to return the local or global ID. Defaults to "local".

        Returns:
        -------
            int: The ID for the token.
        """
        if field not in self.token_to_ids:
            raise ValueError(f"Field {field} not in vocabulary.")

        lookup_token = token if token in self.token_to_ids[field] else self.unk_token
        token_ids = self.token_to_ids[field][lookup_token]
        if return_id == "local":
            return token_ids[1]
        elif return_id == "global":
            return token_ids[0]
        raise ValueError(f"Invalid return_id {return_id}")

    def get_token(self, field: str, token_id: int, return_id="local") -> str:
        """
        Gets the token for a given ID and field key.

        Args:
        ----
            field (str): The key for the field.
            token_id (int): The ID to get the token for.
            return_id (str, optional): Whether to return the local or global ID. Defaults to "local".
        """
        if return_id == "local":
            return self.get_local_token(field, token_id)
        if return_id == "global":
            return self.get_global_token(token_id)
        raise ValueError(f"Invalid return_id {return_id}")

    def get_local_token(self, field: str, local_token_id: int) -> str:
        """
        Gets the token for a given ID and field key.

        Args:
        ----
            field (str): The key for the field.
            local_token_id (int): The local ID to get the token for.

        Returns:
        -------
            str: The token for the ID.
        """
        if field not in self.token_to_ids:
            raise ValueError(f"Field {field} not in vocabulary.")
        return self.local_id_to_token[field].get(local_token_id, self.unk_token)

    def get_global_token(self, global_id: int) -> str:
        """
        Gets the token for a given ID and field key.

        Args:
        ----
            global_id (int): The global ID to get the token for.

        Returns:
        -------
            str: The token for the ID.
        """
        if global_id in self.global_id_to_token:
            return self.global_id_to_token[global_id][0]
        else:
            raise ValueError(f"Global ID {global_id} not in vocabulary.")

    def get_field_specific_token_ids(
        self, field: str, return_id: str = "local"
    ) -> list[int]:
        """
        Gets the IDs for a given field.

        Args:
        ----
            field (str): The key for the field.
            return_id (str, optional): The type of ID to return. Defaults to "local".

        Returns:
        -------
            list[int]: The IDs for the field.
        """
        if field not in self.token_to_ids:
            raise ValueError(f"Field {field} not in vocabulary.")
        if return_id == "local":
            return [
                self.token_to_ids[field][token][1] for token in self.token_to_ids[field]
            ]
        elif return_id == "global":
            return [
                self.token_to_ids[field][token][0] for token in self.token_to_ids[field]
            ]
        raise ValueError(f"Invalid return_id {return_id}")

    def get_local_ids_from_global_ids(
        self,
        global_ids: list[int],
    ) -> list[int]:
        """
        Gets the local IDs for a given list of global IDs.

        Args:
        ----
            global_ids (list[int]): The global IDs to get the local IDs for.

        Returns:
        -------
            list[int]: The local IDs for the global IDs.
        """
        local_ids = []
        for global_id in global_ids:
            if global_id not in self.global_id_to_token:
                raise ValueError(f"Global ID {global_id} not in vocabulary.")
            local_ids.append(self.global_id_to_token[global_id][2])
        return local_ids

    def __len__(self) -> int:
        """
        Gets the length of the vocabulary.

        Returns
        -------
            int: The length of the vocabulary.
        """
        return len(self.global_id_to_token)

    def __str__(self) -> str:
        """
        Gets the string representation of the vocabulary.

        Returns
        -------
            str: The string representation of the vocabulary.
        """
        return str(self.global_id_to_token)

    def get_field_specific_tokens(self, field: str) -> list[str]:
        """
        Gets the tokens for a given field.

        Args:
        ----
            field (str): The key for the field.

        Returns:
        -------
            list[str]: The tokens for the field.
        """
        if field not in self.token_to_ids:
            raise ValueError(f"Field {field} not in vocabulary.")
        return list(self.token_to_ids[field].keys())

    def get_fields(self) -> list[str]:
        """
        Gets the fields in the vocabulary.

        Args:
        ----
            exclude_special (bool, optional): Whether to exclude the special fields. Defaults to False.

        Returns:
        -------
                list[str]: The fields in the vocabulary.
        """
        return list(self.token_to_ids.keys())

    @classmethod
    def load(cls, vocab_dir: Path | str) -> "MultiFieldVocabulary":
        """
        Loads the vocabulary from a directory.

        Args:
        ----
            vocab_dir (Union[Path, str]): The directory to load the vocabulary from.

        Returns:
        -------
            MultiFieldVocabulary: The loaded vocabulary.
        """
        multifield_vocab = cls()
        for file in sorted(Path(vocab_dir).glob("*_vocab.txt")):
            field = file.name.rstrip("_vocab.txt")
            with open(file) as f:
                tokens = f.read().splitlines()
                multifield_vocab.add_tokens_to_vocab(field, tokens)
        return multifield_vocab

    @classmethod
    def load_from_dict(
        cls, field_to_tokens: dict[str, list[str]]
    ) -> "MultiFieldVocabulary":
        """
        Loads the vocabulary from a directory.

        Args:
        ----
            vocab_dir (Union[Path, str]): The directory to load the vocabulary from.

        Returns:s
            MultiFieldVocabulary: The loaded vocabulary.
        """
        multifield_vocab = cls()
        for field, tokens in field_to_tokens.items():
            multifield_vocab.add_tokens_to_vocab(field, tokens)
        return multifield_vocab

    @classmethod
    def load_from_json(cls, json_path: Path | str, **kwargs) -> "MultiFieldVocabulary":
        """
        Loads the vocabulary from a JSON file.

        Args:
        ----
            json_path (Union[Path, str]): The path to the JSON file.
            kwargs: other kwargs to pass to initializer

        Returns:
        -------
            MultiFieldVocabulary: The loaded vocabulary.
        """
        with open(json_path) as f:
            data = json.load(f)
        multifield_vocab = cls(**kwargs)
        for field, tokens in data.items():
            multifield_vocab.add_tokens_to_vocab(field, tokens)
        return multifield_vocab

    def save(self, vocab_dir: str | Path):
        """
        Saves the vocabulary to a directory.

        Args:
        ----
            vocab_dir (Union[Path, str]): The directory to save the vocabulary to.
        """
        os.makedirs(vocab_dir, exist_ok=True)
        for field in self.get_fields():
            all_field_tokens = "\n".join(self.token_to_ids[field].keys())
            (Path(vocab_dir) / f"{field}_vocab.txt").write_text(all_field_tokens)

    def save_to_json(
        self, save_directory: str | Path, filename_prefix: str | None = None
    ) -> tuple[str]:
        """
        Saves the vocabulary to a JSON file.

        Args:
        ----
            save_directory (Union[Path, str]): The directory to save the vocabulary to.

        Returns:
        -------
            tuple[str]: The paths to the saved files.
        """
        file_name = f"{filename_prefix}_vocab.json" if filename_prefix else "vocab.json"
        json_path = Path(save_directory) / file_name
        data = {}
        for field in self.get_fields():
            data[field] = list(self.token_to_ids[field].keys())
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)
        return (file_name,)
