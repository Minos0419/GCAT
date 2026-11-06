"""Small dictionary wrapper used by the training loop."""

from __future__ import annotations

import warnings


class XDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num = self[next(iter(self))].shape[0]

    def size(self):
        warnings.warn("Deprecated function: XDict.size().", DeprecationWarning)
        return self._num
