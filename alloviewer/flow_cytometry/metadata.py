from pathlib import Path
from typing import Union

import pandas as pd
from pandas.io.parsers.readers import TextFileReader


class Metadata:
    """Metadata table for sample files and roles.

    The metadata file must contain at least two columns: ``"file_name"`` and
    ``"role"``. Valid roles are ``"NC"``, ``"PC"``, and ``"SAMPLE"``.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to a delimited metadata file. The delimiter is inferred
        automatically.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
    ) -> None:
        df = self.read_file(file_path)
        self._validate_input(df)
        self.df = df

    def __repr__(self) -> str:
        """Return a compact metadata summary.

        Returns
        -------
        str
            Summary containing number of samples and role counts.
        """
        return (
            f"{self.__class__.__name__}("
            f"n_samples={self.df.shape[0]}, "
            f"roles={self.df['role'].value_counts().to_dict()})"
        )

    def read_file(
        self,
        file_path: Union[str, Path],
    ) -> pd.DataFrame:
        """Read a metadata file into a DataFrame.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the metadata file.

        Returns
        -------
        pandas.DataFrame
            Parsed metadata table.
        """
        delim = self._fetch_delimiter(file_path)
        return pd.read_csv(file_path, sep=delim)

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate metadata column names and role values.

        Parameters
        ----------
        df : pandas.DataFrame
            Metadata table.

        Raises
        ------
        ValueError
            If required columns are missing or invalid roles are present.
        """
        self._validate_colnames(df)
        self._validate_rolenames(df)
        return

    def _validate_colnames(self, df: pd.DataFrame) -> None:
        """Validate required metadata columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Metadata table.

        Raises
        ------
        ValueError
            If ``"file_name"`` or ``"role"`` is missing.
        """
        needed_columns = ["file_name", "role"]
        df_cols = df.columns.tolist()

        if any(k not in df_cols for k in needed_columns):
            raise ValueError(
                f"Columns {needed_columns} must be present in metadata file."
            )

        return

    def _validate_rolenames(self, df: pd.DataFrame) -> None:
        """Validate metadata role names.

        Parameters
        ----------
        df : pandas.DataFrame
            Metadata table.

        Raises
        ------
        ValueError
            If any role is not ``"NC"``, ``"PC"``, or ``"SAMPLE"``.
        """
        allowed_roles = ["NC", "PC", "SAMPLE"]
        present_roles = df["role"].unique()

        if any(k not in allowed_roles for k in present_roles):
            raise ValueError("Roles must be 'NC', 'PC' or 'SAMPLE'.")

    def get_df(self) -> pd.DataFrame:
        """Return the metadata table.

        Returns
        -------
        pandas.DataFrame
            Metadata table.
        """
        return self.df

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "Metadata":
        """Create metadata from an existing DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Metadata table with required columns and valid role names.

        Returns
        -------
        Metadata
            Metadata object containing ``df``.

        Raises
        ------
        ValueError
            If required columns are missing or invalid roles are present.
        """
        obj = cls.__new__(cls)
        obj._validate_input(df)
        obj.df = df
        return obj

    def _fetch_delimiter(
        self,
        file: Union[str, Path],
    ) -> str:
        """Infer the delimiter of a metadata file.

        Parameters
        ----------
        file : str or pathlib.Path
            Path to the metadata file.

        Returns
        -------
        str
            Detected delimiter.
        """
        reader: TextFileReader = pd.read_csv(
            file,
            sep=None,
            iterator=True,
            engine="python",
        )

        return reader._engine.data.dialect.delimiter
