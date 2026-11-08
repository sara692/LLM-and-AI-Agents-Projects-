from __future__ import annotations

import io
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Settings, get_settings

logger = logging.getLogger("agent.datasets")

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def is_text_series(series: pd.Series) -> bool:
    """True for text columns under both the classic `object` dtype and pandas 3's `str` dtype."""
    return bool(pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))


def detect_protected_columns(columns, patterns) -> list[str]:
    return [c for c in columns if any(p in str(c).lower() for p in patterns)]


@dataclass
class Dataset:
    id: str
    name: str
    df: pd.DataFrame
    protected_columns: list[str] = field(default_factory=list)
    known_tokens: frozenset[str] = frozenset()      # e.g. {"Omar", "Hassan"}
    known_full_names: frozenset[str] = frozenset()  # e.g. {"omar hassan"}

    @classmethod
    def from_dataframe(
        cls, id: str, df: pd.DataFrame, patterns=None, name: str | None = None
    ) -> "Dataset":
        patterns = tuple(patterns) if patterns is not None else get_settings().protected_patterns
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        protected = detect_protected_columns(df.columns, patterns)

        tokens: set[str] = set()
        full: set[str] = set()
        for col in protected:
            if not is_text_series(df[col]):
                continue
            for value in df[col].dropna().astype(str).unique()[:50_000]:
                value = value.strip()
                if not value:
                    continue
                full.add(value.lower())
                tokens.update(_TOKEN.findall(value))
        return cls(
            id=id, name=name or id, df=df, protected_columns=protected,
            known_tokens=frozenset(tokens), known_full_names=frozenset(full),
        )

    @property
    def visible_columns(self) -> list[str]:
        """Columns the LLM and the user's clarification prompts may see."""
        hidden = set(self.protected_columns)
        return [c for c in self.df.columns if c not in hidden]

    def public_schema(self) -> list[dict]:
        """Column metadata for the API/UI (names + dtypes only, no data)."""
        return [
            {
                "name": c,
                "dtype": str(self.df[c].dtype),
                "protected": c in self.protected_columns,
            }
            for c in self.df.columns
        ]

    def llm_schema(self, include_values: bool = True, max_values: int = 12) -> dict:
        """Schema handed to the LLM: visible columns, dtypes, low-cardinality labels."""
        columns, dtypes, categories = [], {}, {}
        for c in self.visible_columns:
            columns.append(c)
            dtypes[c] = str(self.df[c].dtype)
            if include_values and is_text_series(self.df[c]):
                uniques = self.df[c].dropna().unique()
                if 0 < len(uniques) <= max_values:
                    categories[c] = sorted(str(u) for u in uniques)
        schema: dict = {"columns": columns, "dtypes": dtypes}
        if categories:
            schema["category_values"] = categories
        return schema


class DatasetStore:
    """In-memory registry of datasets (default files + user uploads)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._items: dict[str, Dataset] = {}
        self._uploads: list[str] = []
        self._lock = threading.Lock()

    # -- loading -----------------------------------------------------------
    def load_directory(self, directory: Path | None = None) -> None:
        directory = Path(directory or self.settings.data_dir)
        for path in sorted(directory.glob("*.csv")):
            try:
                df = pd.read_csv(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s: %s", path.name, exc)
                continue
            self.add_dataframe(path.stem, df, name=path.stem.replace("_", " ").title())

    def add_dataframe(self, dataset_id: str, df: pd.DataFrame, name: str | None = None) -> Dataset:
        ds = Dataset.from_dataframe(dataset_id, df, self.settings.protected_patterns, name)
        with self._lock:
            self._items[dataset_id] = ds
        logger.info(
            "Loaded dataset '%s': %d rows, %d columns, protected=%s",
            dataset_id, len(ds.df), len(ds.df.columns), ds.protected_columns,
        )
        return ds

    def add_upload(self, filename: str, content: bytes) -> Dataset:
        limit = self.settings.max_upload_mb * 1024 * 1024
        if len(content) > limit:
            raise ValueError(f"File is larger than {self.settings.max_upload_mb} MB.")
        suffix = Path(filename).suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(io.BytesIO(content))
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise ValueError("Only .csv, .xlsx and .xls files are supported.")
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not read the file: {type(exc).__name__}") from exc

        if df.empty or len(df.columns) == 0:
            raise ValueError("The file contains no data.")
        if len(df.columns) > 200 or len(df) > 500_000:
            raise ValueError("Dataset is too large (max 200 columns / 500,000 rows).")
        if pd.Index([str(c).strip() for c in df.columns]).has_duplicates:
            raise ValueError("Duplicate column names are not supported.")

        slug = re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower()).strip("_") or "dataset"
        dataset_id, n = f"upload_{slug}", 1
        with self._lock:
            while dataset_id in self._items:
                n += 1
                dataset_id = f"upload_{slug}_{n}"
            if len(self._uploads) >= self.settings.max_uploaded_datasets:
                oldest = self._uploads.pop(0)  # evict the oldest upload
                self._items.pop(oldest, None)
            self._uploads.append(dataset_id)
        return self.add_dataframe(dataset_id, df, name=Path(filename).stem)

    # -- access ------------------------------------------------------------
    def get(self, dataset_id: str | None = None) -> Dataset:
        with self._lock:
            if dataset_id is None:
                if not self._items:
                    raise KeyError("No datasets are loaded.")
                dataset_id = self.default_id
            if dataset_id not in self._items:
                raise KeyError(dataset_id)
            return self._items[dataset_id]

    @property
    def default_id(self) -> str:
        for preferred in ("employees", "sales_dataset"):
            if preferred in self._items:
                return preferred
        return next(iter(self._items))

    def list(self) -> list[Dataset]:
        with self._lock:
            return list(self._items.values())
