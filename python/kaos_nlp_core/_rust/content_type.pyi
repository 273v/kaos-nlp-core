"""Type stubs for kaos_nlp_core._rust.content_type."""

from typing import TypedDict

class _ContentTypeDict(TypedDict):
    mime_type: str
    extension: str
    group: str

def py_detect(data: bytes) -> _ContentTypeDict: ...
