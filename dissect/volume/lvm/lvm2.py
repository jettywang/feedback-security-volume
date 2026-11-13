from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, BinaryIO

from dissect.util import ts

from dissect.volume.lvm.metadata import VolumeGroup
from dissect.volume.lvm.physical import LVM2Device
from dissect.volume.lvm.selector import MetadataSelection, select_metadata

if TYPE_CHECKING:
    from datetime import datetime

log = logging.getLogger(__name__)
log.setLevel(os.getenv("DISSECT_LOG_LVM", "CRITICAL"))


class LVM2:
    """Logical Volume Manager.

    A Volume Group is spread over several separately collected Physical Volumes. The metadata on those physical
    volumes is replicated but the copies are not necessarily from the same generation (each metadata update bumps
    the ``seqno``). On construction every metadata area of every supplied physical volume is therefore validated
    and reconciled by :func:`~dissect.volume.lvm.selector.select_metadata` before a single internally consistent
    configuration is adopted and the actual devices are attached.
    """

    def __init__(self, fh: list[BinaryIO | LVM2Device] | BinaryIO | LVM2Device):
        self.fh = [fh] if not isinstance(fh, list) else fh
        if not self.fh:
            raise ValueError("At least one file-like object is required")

        devices = [LVM2Device(fh) if not isinstance(fh, LVM2Device) else fh for fh in self.fh]
        # Keyed by the on-disk PV id so attachment and lookups never depend on the input ordering.
        self.devices: dict[str, LVM2Device] = {device.id: device for device in devices}

        self.selection: MetadataSelection = select_metadata(devices)
        self.metadata: dict = self.selection.metadata
        self.volume_group: VolumeGroup = self.selection.volume_group

        self.contents: str = self.metadata["contents"]
        self.version: int = self.metadata["version"]
        self.description: str | None = self.metadata.get("description")
        self.creation_host: str | None = self.metadata.get("creation_host")
        self.creation_time: datetime | None = None
        if creation_time := self.metadata.get("creation_time"):
            self.creation_time: datetime | None = ts.from_unix(creation_time)

        # Attach the physical devices that were actually supplied. Referenced PVs that are missing simply remain
        # without a device; opening them raises a clear error instead of silently producing wrong data.
        self.volume_group.attach(self.devices)
        self.missing_physical_volumes: list[str] = [pv.name for pv in self.volume_group.pv if pv.dev is None]
        if self.missing_physical_volumes:
            log.warning(
                "Volume Group %s references physical volume(s) not present in the supplied devices: %s",
                self.volume_group.name,
                ", ".join(self.missing_physical_volumes),
            )

    def __repr__(self) -> str:
        return f"<LVM2 vg={self.vg}>"

    @property
    def vg(self) -> VolumeGroup:
        return self.volume_group

    def metadata_report(self) -> str:
        """Return a human readable report of the adopted and excluded metadata copies and their reasons."""
        return self.selection.report()
