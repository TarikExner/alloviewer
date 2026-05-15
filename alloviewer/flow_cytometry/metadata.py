from pathlib import Path
from typing import Union
import pandas as pd
from pandas.io.parsers.readers import TextFileReader


class Metadata:

    def __init__(self,
                 file_path: Union[str, Path]) -> None:

        df = self.read_file(file_path)
        self._validate_input(df)
        self.df = df

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" +
            f"n_samples={self.df.shape[0]}, " +
            f"roles={self.df['role'].value_counts().to_dict()})"
        )

    def read_file(self,
                  file_path: Union[str, Path]) -> pd.DataFrame:
        delim = self._fetch_delimiter(file_path)
        return pd.read_csv(file_path, sep=delim)

    def _validate_input(self, df: pd.DataFrame) -> None:
        self._validate_colnames(df)
        self._validate_rolenames(df)
        return

    def _validate_colnames(self, df: pd.DataFrame) -> None:
        needed_columns = ["file_name", "role"]
        df_cols = df.columns.tolist()
        if any(k not in df_cols for k in needed_columns):
                raise ValueError(f"Columns {needed_columns} must be present in metadata file.")
        return

    def _validate_rolenames(self, df: pd.DataFrame) -> None:
        allowed_roles = ["NC", "PC", "SAMPLE"]
        present_roles = df["role"].unique()
        if any(k not in allowed_roles for k in present_roles):
            raise ValueError("Roles must be 'NC', 'PC' or 'SAMPLE'.")

    def get_df(self):
        return self.df

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "Metadata":
        obj = cls.__new__(cls)
        obj._validate_input(df)
        obj.df = df
        return obj

    def _fetch_delimiter(self,
                         file: Union[str, Path]) -> str:
        reader: TextFileReader = pd.read_csv(file,
                                             sep = None,
                                             iterator = True,
                                             engine = "python")
        return reader._engine.data.dialect.delimiter




