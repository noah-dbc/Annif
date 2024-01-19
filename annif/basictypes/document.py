from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    text: str
    subject_set: Any | None = None  # should be SubjectSet, but that gives circular import
