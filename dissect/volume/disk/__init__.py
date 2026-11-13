from __future__ import annotations

from dissect.volume.disk.disk import Disk
from dissect.volume.disk.partition import Partition
from dissect.volume.exceptions import DiskError, GPTConflictError

__all__ = [
    "Disk",
    "DiskError",
    "GPTConflictError",
    "Partition",
]
