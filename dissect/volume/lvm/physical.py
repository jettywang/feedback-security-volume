from __future__ import annotations

import ast
import re
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, BinaryIO

from dissect.util.stream import RunlistStream

from dissect.volume.exceptions import LVM2Error
from dissect.volume.lvm.c_lvm2 import LABEL_SCAN_SECTORS, SECTOR_SIZE, c_lvm, lvm_crc32

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class MetadataCopy:
    """A single, fully contained copy of a Volume Group metadata text.

    Each Physical Volume can host several metadata areas, and each metadata area can contain several raw
    locations (metadata copies). This object describes one such raw location together with everything needed
    to decide whether it can be trusted.

    Args:
        device: The Physical Volume this copy was read from.
        metadata_area: The zero-based index of the metadata area on the device.
        copy_index: The zero-based index of the raw location within that metadata area.
        offset: Absolute byte offset of the metadata text on the underlying device.
        size: Size in bytes of the metadata text (as recorded in the raw location descriptor).
        checksum: Checksum stored in the raw location descriptor.
        flags: Raw location flags, e.g. ``RAW_LOCN_IGNORED``.
        text: The raw metadata text, or ``None`` if it could not be read.
        data: The parsed metadata, or ``None`` if it could not be parsed or validated.
        error: Human readable reason this copy is unusable, or ``None`` when valid.
    """

    device: LVM2Device
    metadata_area: int
    copy_index: int
    offset: int
    size: int
    checksum: int
    flags: int
    text: str | None
    data: dict | None
    error: str | None = None

    @property
    def valid(self) -> bool:
        """Whether this copy passed its checksum and could be parsed."""
        return self.error is None and self.data is not None

    @property
    def ignored(self) -> bool:
        """Whether LVM marked this raw location as ignored."""
        return bool(self.flags & c_lvm.RAW_LOCN_IGNORED)

    def __repr__(self) -> str:
        status = "valid" if self.valid else f"invalid({self.error})"
        return (
            f"<MetadataCopy pv={self.device.id} mda={self.metadata_area} copy={self.copy_index} "
            f"size={self.size} {status}>"
        )


class LVM2Device:
    def __init__(self, fh: BinaryIO):
        self.fh = fh

        for i in range(LABEL_SCAN_SECTORS):
            fh.seek(i * SECTOR_SIZE)
            lbl = c_lvm.label_header(fh)
            if lbl.id == b"LABELONE":
                self.label = lbl
                self.label_offset = i * SECTOR_SIZE
                break
        else:
            raise LVM2Error("Can't find physical volume label header")

        fh.seek(self.label_offset + self.label.offset)

        self.header = c_lvm.pv_header(fh)
        self.id = self.header.pv_uuid.rstrip(b"\x00").decode()
        self.size = self.header.device_size

        self._data_area_descriptors = _read_descriptors(fh, c_lvm.disk_locn)

        if len(self._data_area_descriptors) != 1:
            # According to the LVM2 code, there should only be 1 data area descriptor for each Physical Volume
            raise RuntimeError(f"There should be 1 data area descriptor, found {len(self._data_area_descriptors)}")

        self._metadata_area_descriptors = _read_descriptors(fh, c_lvm.disk_locn)

        self._metadata_areas = []
        for desc in self._metadata_area_descriptors:
            self.fh.seek(desc.offset)
            header = c_lvm.mda_header(self.fh)
            raw_location_descriptors = _read_descriptors(self.fh, c_lvm.raw_locn)
            self._metadata_areas.append((header, raw_location_descriptors))

        self._data_area_offsets = []

        offset = 0
        for area in self._data_area_descriptors:
            if offset != 0:
                self._data_area_offsets.append(offset)
            offset += area.size

    def __repr__(self) -> str:
        return f"<LVMDevice id={self.id}, size={self.size:#x}>"

    @cached_property
    def metadata_copies(self) -> list[MetadataCopy]:
        """Read and validate every metadata copy on this Physical Volume.

        All metadata areas and their raw locations are inspected independently. A copy is only reported as
        :attr:`MetadataCopy.valid` when its raw location is not marked as ignored, its checksum verifies and its
        text can be parsed. Invalid copies are still returned so callers can document why they were rejected.
        """
        copies: list[MetadataCopy] = []

        for area_index, (mda_header, raw_locations) in enumerate(self._metadata_areas):
            for copy_index, raw_locn in enumerate(raw_locations):
                offset = mda_header.start + raw_locn.offset
                size = raw_locn.size

                copy = MetadataCopy(
                    device=self,
                    metadata_area=area_index,
                    copy_index=copy_index,
                    offset=offset,
                    size=size,
                    checksum=raw_locn.checksum,
                    flags=raw_locn.flags,
                    text=None,
                    data=None,
                )

                if raw_locn.flags & c_lvm.RAW_LOCN_IGNORED:
                    copy.error = "raw location is marked as ignored"
                    copies.append(copy)
                    continue

                if size == 0:
                    copy.error = "raw location has no metadata"
                    copies.append(copy)
                    continue

                self.fh.seek(offset)
                raw_text = self.fh.read(size)
                # The checksum protects the full raw location including its trailing NUL terminator, but the
                # text parser expects the NUL to be removed.
                copy.text = raw_text.rstrip(b"\x00").decode(errors="replace")

                actual_checksum = lvm_crc32(raw_text)
                if actual_checksum != raw_locn.checksum:
                    copy.error = (
                        f"checksum mismatch: expected {raw_locn.checksum:#010x}, calculated {actual_checksum:#010x}"
                    )
                    copies.append(copy)
                    continue

                try:
                    copy.data = parse_metadata(copy.text)
                except Exception as err:
                    copy.error = f"metadata could not be parsed: {err}"
                copies.append(copy)

        return copies

    @cached_property
    def metadata(self) -> dict | None:
        """Return the first valid metadata copy on this device, or ``None`` if there is no valid copy.

        Kept for backwards compatibility. New code should prefer :attr:`metadata_copies` so all available
        copies and the reasons invalid ones were rejected can be inspected.
        """
        for copy in self.metadata_copies:
            if copy.valid:
                return copy.data
        return None

    def read_sectors(self, sector: int, count: int) -> bytes:
        r = []
        area_idx = bisect_right(self._data_area_offsets, sector * SECTOR_SIZE)

        while count > 0:
            area = self._data_area_descriptors[area_idx]

            area_size = area.size or self.size
            area_remaining_sectors = (area_size // SECTOR_SIZE) - (sector - area.offset // SECTOR_SIZE)
            read_sectors = min(area_remaining_sectors, count)

            self.fh.seek(area.offset + sector * SECTOR_SIZE)
            r.append(self.fh.read(read_sectors * SECTOR_SIZE))
            sector += read_sectors
            count -= read_sectors

            area_idx += 1

        return b"".join(r)

    def open(self) -> BinaryIO:
        # Only accounts for one data area descriptor
        # When multiple descriptors get allowed in the future, this code should be adjusted to reflect that
        area = self._data_area_descriptors[0]
        runlist = [(area.offset // SECTOR_SIZE, self.size // SECTOR_SIZE)]
        return RunlistStream(self.fh, runlist, self.size, SECTOR_SIZE)


def _read_descriptors(
    fh: BinaryIO, ctype: type[c_lvm.disk_locn | c_lvm.raw_locn]
) -> list[c_lvm.disk_locn | c_lvm.raw_locn]:
    # Read an unspecified number of descriptors from the file handle, stopping on all zeros
    return ctype[None](fh)


def parse_metadata(string: str) -> dict:
    root = {}
    current = root
    parents = {}

    s = re.sub(r"(#[^\"]+?)$", "", string, flags=re.MULTILINE)

    it = iter(s.split("\n"))
    for line in it:
        line = line.strip()
        if not line or line[0] == "#":
            continue

        if line[-1] == "{":
            name = line[:-1].strip()

            child = {}
            parent = current
            parents[id(child)] = parent
            parent[name] = child
            current = child
            continue

        if line[-1] == "}":
            current = parents[id(current)]
            continue

        k, v = _parse_key_value(line, it)
        current[k] = v

    root["_raw"] = string
    return root


def _parse_key_value(s: str, it: Iterator[str]) -> tuple[str, Any]:
    key, value = s.strip().split("=", 1)
    key = key.strip()
    value = value.strip()

    if value[0] == "[" and value[-1] != "]":
        lines = [value]
        for line in it:
            lines.append(line)
            if line[-1] == "]":
                break
        value = "".join(lines)

    value = ast.literal_eval(value)

    return key, value
