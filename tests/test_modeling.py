"""Regression coverage for model-artifact provenance."""

from __future__ import annotations

from io import BytesIO

from ast_asr.modeling import directory_content_hash


class _RelativeKey:
    def __init__(self, native_key: str) -> None:
        self._native_key = native_key

    def as_posix(self) -> str:
        return self._native_key.replace("\\", "/")


class _FixturePath:
    def __init__(self, native_key: str, contents: bytes, *, ordering: str) -> None:
        self._native_key = native_key
        self._contents = contents
        self._ordering = ordering

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _FixturePath):
            return NotImplemented
        if self._ordering == "windows":
            return self._native_key.casefold() < other._native_key.casefold()
        return self._native_key < other._native_key

    def is_file(self) -> bool:
        return True

    def relative_to(self, _directory: object) -> _RelativeKey:
        return _RelativeKey(self._native_key)

    def open(self, _mode: str) -> BytesIO:
        return BytesIO(self._contents)


class _FixtureDirectory:
    def __init__(self, *, ordering: str) -> None:
        self._paths = [
            _FixturePath("adapter_config.json", b"adapter", ordering=ordering),
            _FixturePath("README.md", b"readme", ordering=ordering),
            _FixturePath("processor\\config.json", b"processor", ordering=ordering),
        ]

    def rglob(self, _pattern: str) -> list[_FixturePath]:
        return list(reversed(self._paths))


def test_directory_hash_is_identical_for_windows_and_posix_path_ordering() -> None:
    """Native path ordering must not change a checkpoint revision."""
    posix_hash = directory_content_hash(_FixtureDirectory(ordering="posix"))  # type: ignore[arg-type]
    windows_hash = directory_content_hash(_FixtureDirectory(ordering="windows"))  # type: ignore[arg-type]

    assert posix_hash == windows_hash
