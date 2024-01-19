from dataclasses import dataclass


@dataclass(frozen=True)
class Subject:
    uri: str
    labels: dict
    notation: str
