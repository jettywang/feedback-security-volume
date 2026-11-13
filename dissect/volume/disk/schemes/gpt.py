from __future__ import annotations

import io
import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID

from dissect.cstruct import cstruct

from dissect.volume.disk.partition import Partition
from dissect.volume.disk.schemes.mbr import MBR
from dissect.volume.exceptions import DiskError, GPTConflictError

if TYPE_CHECKING:
    from collections.abc import Iterator

gpt_def = """
// http://en.wikipedia.org/wiki/GUID_Partition_Table
struct GPT_HEADER {
    char        signature[8];
    uint32      revision;
    uint32      header_size;
    uint32      crc32;
    uint32      reserved;
    uint64      current_lba;
    uint64      backup_lba;
    uint64      first_usable_lba;
    uint64      last_usable_lba;
    char        guid[16];
    uint64      lba_partition_array;
    uint32      partition_table_count;
    uint32      partition_entry_size;
    uint32      partition_table_crc;
    char        _[420];
};

struct GPT_PARTITION {
    char        type_guid[16];
    char        partition_guid[16];
    uint64      first_lba;
    uint64      last_lba;
    uint64      attribute_flags;
    char        name[72];   // UTF16 encoded
};

// 0 (0x00)     16 bytes     Partition type GUID
// 16 (0x10)    16 bytes     Unique partition GUID
// 32 (0x20)    8 bytes      First LBA (little endian)
// 40 (0x28)    8 bytes      Last LBA (inclusive, usually odd)
// 48 (0x30)    8 bytes      Attribute flags (e.g. bit 60 denotes read-only)
// 56 (0x38)    72 bytes     Partition name (36 UTF-16LE code units)
"""

c_gpt = cstruct().load(gpt_def)

GPT_SIGNATURE = b"EFI PART"
GPT_HEADER_SIZE = 92
GPT_HEADER_STRUCT_SIZE = 512
GPT_PARTITION_SIZE = 128


@dataclass
class GPTCopy:
    """The validation state of one GPT header and its partition array.

    Both copies are retained after parsing, including copies that cannot be used, so the
    evidence from the rejected copy remains available for inspection.

    Args:
        source: Either ``"primary"`` or ``"backup"``.
    """

    source: str
    offset: int | None = None
    header: c_gpt.GPT_HEADER | None = None
    header_valid: bool = False
    array_valid: bool = False
    entries_valid: bool = False
    geometry_valid: bool = False
    error: str | None = None
    array_error: str | None = None
    array_offset: int | None = None
    entries: list[c_gpt.GPT_PARTITION] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Whether the header, partition array checksum and disk geometry all validated."""
        return self.header_valid and self.array_valid and self.entries_valid and self.geometry_valid

    @property
    def usable(self) -> bool:
        """Whether the copy can be used, even if the partition array checksum failed."""
        return self.header_valid and self.entries_valid and self.geometry_valid

    @property
    def status(self) -> str:
        """Either ``"valid"`` or ``"invalid"``."""
        return "valid" if self.valid else "invalid"

    @property
    def reason(self) -> str | None:
        """The first validation failure, if any."""
        return self.error

    @property
    def partition_ranges(self) -> list[tuple[int, int, int]]:
        """The ``(index, first_lba, last_lba)`` tuples of all used partition entries."""
        return [
            (index, entry.first_lba, entry.last_lba)
            for index, entry in enumerate(self.entries)
            if entry.type_guid != b"\x00" * 16 and entry.first_lba != 0
        ]

    def as_dict(self) -> dict[str, object]:
        """Return a serializable summary of this copy."""
        return {
            "source": self.source,
            "status": self.status,
            "valid": self.valid,
            "header_valid": self.header_valid,
            "array_valid": self.array_valid,
            "entries_valid": self.entries_valid,
            "geometry_valid": self.geometry_valid,
            "offset": self.offset,
            "array_offset": self.array_offset,
            "error": self.error,
            "array_error": self.array_error,
            "warnings": list(self.warnings),
        }

    def __repr__(self) -> str:
        return f"<GPTCopy source={self.source!r} status={self.status!r} offset={self.offset!r} error={self.error!r}>"


class GPT:
    """GUID Partition Table.

    Both the primary and the backup GPT copy are read and validated. A damaged primary
    header or partition array does not discard the disk when the backup copy is usable.

    The selected copy is recorded in :attr:`selected` and :attr:`selected_source`. The
    other copy, including its validation status and errors, remains available through
    :attr:`other`. The source stream is only ever read, never written, so the original
    image is not repaired.

    When both copies are fully valid but describe different layouts,
    :class:`~dissect.volume.exceptions.GPTConflictError` is raised. Pass an explicit
    ``source`` to select one of them for analysis.

    Args:
        fh: File-like object of the disk.
        sector_size: Sector size in bytes.
        source: Explicitly select ``"primary"`` or ``"backup"``. Required to resolve a
            conflict between two fully valid copies.
        preferred_source: Alias for ``source``.
    """

    def __init__(
        self,
        fh: BinaryIO,
        sector_size: int = 512,
        *,
        source: str | None = None,
        preferred_source: str | None = None,
    ):
        self.fh = fh
        self.sector_size = sector_size

        requested_source = source or preferred_source
        if requested_source is not None:
            requested_source = requested_source.lower()
            if requested_source not in ("primary", "backup"):
                raise ValueError(f"Invalid GPT source {requested_source!r}, expected 'primary' or 'backup'")

        self.mbr = MBR(fh, sector_size=sector_size)
        gpt_parts = [p for p in self.mbr.partitions if p.type == 0xEE]
        if not gpt_parts:
            raise DiskError("Invalid protective/hybrid MBR, could not find 0xEE GPT partition")

        self.primary_lba = gpt_parts[0].offset // sector_size
        self.disk_last_lba = self._disk_last_lba()
        self.sector_count = self.disk_last_lba + 1 if self.disk_last_lba is not None else None

        self.primary = self._read_copy("primary", self.primary_lba)
        self.backup = self._read_copy("backup", self._backup_lba())
        self.copies = {"primary": self.primary, "backup": self.backup}

        self._cross_check()
        self._select(requested_source)

        self.partitions: list[Partition] = list(self._partitions())

    def _disk_last_lba(self) -> int | None:
        try:
            position = self.fh.tell()
            self.fh.seek(0, io.SEEK_END)
            size = self.fh.tell()
            self.fh.seek(position)
        except (AttributeError, OSError, ValueError):
            return None

        if not size or size < self.sector_size:
            return None
        return (size // self.sector_size) - 1

    def _backup_lba(self) -> int | None:
        candidates = []

        if self.primary.header_valid:
            candidates.append(self.primary.header.backup_lba)
        if self.disk_last_lba is not None and self.disk_last_lba not in candidates:
            candidates.append(self.disk_last_lba)

        if not candidates:
            return None

        for lba in candidates:
            if lba < 0:
                continue
            self.fh.seek(lba * self.sector_size)
            if self.fh.read(8) == GPT_SIGNATURE:
                return lba

        # No header was found at any candidate location. Prefer the location the primary
        # header declared when it lies inside the image; otherwise report against the last
        # sector of the image we have.
        for lba in candidates:
            if lba >= 0 and (self.disk_last_lba is None or lba <= self.disk_last_lba):
                return lba
        return candidates[0]

    def _read_copy(self, source: str, location_lba: int | None) -> GPTCopy:
        copy = GPTCopy(source=source)
        if location_lba is None or location_lba < 0:
            copy.error = f"Unable to determine {source} GPT header location"
            return copy

        offset = location_lba * self.sector_size
        copy.offset = offset

        self.fh.seek(offset)
        raw = self.fh.read(self.sector_size)
        if len(raw) < GPT_HEADER_SIZE:
            copy.error = f"No {source} GPT header found at LBA {location_lba} (short read)"
            return copy

        if raw[:8] != GPT_SIGNATURE:
            copy.error = f"Invalid {source} GPT signature at LBA {location_lba}, got {raw[:8]!r}"
            return copy

        header = c_gpt.GPT_HEADER(io.BytesIO(raw))
        copy.header = header

        if header.header_size < GPT_HEADER_SIZE or header.header_size > GPT_HEADER_STRUCT_SIZE:
            copy.error = (
                f"Invalid {source} GPT header size {header.header_size}, "
                f"expected between {GPT_HEADER_SIZE} and {GPT_HEADER_STRUCT_SIZE}"
            )
            return copy

        if header.header_size > len(raw):
            copy.error = f"Short {source} GPT header: need {header.header_size} bytes, got {len(raw)}"
            return copy

        header_crc = zlib.crc32(raw[:16] + b"\x00\x00\x00\x00" + raw[20 : header.header_size]) & 0xFFFFFFFF
        if header_crc != header.crc32:
            copy.error = f"Invalid {source} GPT header checksum, expected 0x{header.crc32:08x}, got 0x{header_crc:08x}"
            return copy
        copy.header_valid = True

        if header.reserved != 0:
            copy.header_valid = False
            copy.error = f"Invalid {source} GPT header reserved field, expected 0, got 0x{header.reserved:x}"
            return copy

        if header.current_lba != location_lba:
            copy.header_valid = False
            copy.error = (
                f"{source.capitalize()} GPT header says current_lba={header.current_lba}, "
                f"but was found at LBA {location_lba}"
            )
            return copy

        if source == "backup" and header.backup_lba != self.primary_lba:
            copy.header_valid = False
            copy.error = (
                f"Backup GPT header points to primary LBA {header.backup_lba}, "
                f"but the protective MBR points to LBA {self.primary_lba}"
            )
            return copy

        if header.backup_lba == header.current_lba:
            copy.header_valid = False
            copy.error = f"{source.capitalize()} GPT header alternate LBA equals its current LBA"
            return copy

        if header.first_usable_lba > header.last_usable_lba:
            copy.header_valid = False
            copy.error = (
                f"Invalid {source} GPT usable range: first_usable_lba={header.first_usable_lba} > "
                f"last_usable_lba={header.last_usable_lba}"
            )
            return copy

        if header.first_usable_lba <= self.primary_lba:
            copy.header_valid = False
            copy.error = (
                f"Invalid {source} GPT usable range starts at LBA {header.first_usable_lba}, "
                f"at or before the primary header LBA {self.primary_lba}"
            )
            return copy

        if source == "primary":
            if header.last_usable_lba >= header.backup_lba:
                copy.header_valid = False
                copy.error = (
                    f"Invalid primary GPT usable range ends at LBA {header.last_usable_lba}, "
                    f"at or after the backup header LBA {header.backup_lba}"
                )
                return copy
        elif header.last_usable_lba >= header.current_lba:
            copy.header_valid = False
            copy.error = (
                f"Invalid backup GPT usable range ends at LBA {header.last_usable_lba}, "
                f"at or after the backup header LBA {header.current_lba}"
            )
            return copy

        if header.partition_table_count == 0 or header.partition_entry_size < GPT_PARTITION_SIZE:
            copy.header_valid = False
            copy.error = (
                f"Invalid {source} GPT partition array geometry: count={header.partition_table_count}, "
                f"entry_size={header.partition_entry_size}"
            )
            return copy

        if header.partition_entry_size % 8 != 0:
            copy.header_valid = False
            copy.error = f"Invalid {source} GPT partition entry size {header.partition_entry_size}"
            return copy

        array_size = header.partition_table_count * header.partition_entry_size
        array_sectors = (array_size + self.sector_size - 1) // self.sector_size
        array_end_lba = header.lba_partition_array + array_sectors

        if source == "primary":
            if header.lba_partition_array <= header.current_lba:
                copy.geometry_valid = False
                copy.error = (
                    f"Invalid primary GPT partition array LBA {header.lba_partition_array}, "
                    f"at or before header LBA {header.current_lba}"
                )
                return copy
            if array_end_lba > header.first_usable_lba:
                copy.geometry_valid = False
                copy.error = (
                    f"Primary GPT partition array ends at LBA {array_end_lba}, "
                    f"after first usable LBA {header.first_usable_lba}"
                )
                return copy
        else:
            if header.lba_partition_array <= header.last_usable_lba:
                copy.geometry_valid = False
                copy.error = (
                    f"Backup GPT partition array LBA {header.lba_partition_array}, "
                    f"at or before last usable LBA {header.last_usable_lba}"
                )
                return copy
            if array_end_lba > header.current_lba:
                copy.geometry_valid = False
                copy.error = (
                    f"Backup GPT partition array ends at LBA {array_end_lba}, "
                    f"after backup header LBA {header.current_lba}"
                )
                return copy

        copy.array_offset = header.lba_partition_array * self.sector_size

        if self.disk_last_lba is not None:
            if array_end_lba > self.disk_last_lba + 1:
                copy.geometry_valid = False
                copy.error = (
                    f"{source.capitalize()} GPT partition array ends at LBA {array_end_lba}, "
                    f"beyond the disk geometry ({self.sector_count} sectors)"
                )
                return copy
            if header.current_lba > self.disk_last_lba:
                copy.geometry_valid = False
                copy.error = (
                    f"{source.capitalize()} GPT header LBA {header.current_lba} is beyond the disk geometry "
                    f"({self.sector_count} sectors)"
                )
                return copy

        self.fh.seek(copy.array_offset)
        array_raw = self.fh.read(array_size)
        if len(array_raw) < array_size:
            copy.array_valid = False
            copy.array_error = f"Short {source} GPT partition array: expected {array_size} bytes, got {len(array_raw)}"
            copy.geometry_valid = False
            copy.error = copy.array_error
            return copy

        try:
            copy.entries = self._parse_entries(source, array_raw, header)
        except ValueError as e:
            copy.array_error = str(e)
            copy.error = str(e)
            return copy

        copy.entries_valid = True
        copy.geometry_valid = True

        array_crc = zlib.crc32(array_raw) & 0xFFFFFFFF
        if array_crc != header.partition_table_crc:
            copy.array_valid = False
            copy.array_error = (
                f"Invalid {source} GPT partition array checksum, "
                f"expected 0x{header.partition_table_crc:08x}, got 0x{array_crc:08x}"
            )
            copy.error = copy.array_error
            return copy

        copy.array_valid = True
        copy.error = None
        return copy

    def _parse_entries(self, source: str, array_raw: bytes, header: c_gpt.GPT_HEADER) -> list[c_gpt.GPT_PARTITION]:
        entries = []
        for part_num in range(header.partition_table_count):
            entry_offset = part_num * header.partition_entry_size
            entry_raw = array_raw[entry_offset : entry_offset + GPT_PARTITION_SIZE]
            if len(entry_raw) < GPT_PARTITION_SIZE:
                raise ValueError(f"Short {source} GPT partition entry {part_num + 1}")

            partition = c_gpt.GPT_PARTITION(io.BytesIO(entry_raw))
            if partition.type_guid == b"\x00" * 16:
                entries.append(partition)
                continue

            if partition.first_lba == 0:
                raise ValueError(f"Invalid {source} GPT partition {part_num + 1}: first_lba is zero for a used entry")

            if partition.first_lba > partition.last_lba:
                raise ValueError(
                    f"Invalid {source} GPT partition {part_num + 1} range: "
                    f"first_lba={partition.first_lba} > last_lba={partition.last_lba}"
                )

            if partition.first_lba < header.first_usable_lba or partition.last_lba > header.last_usable_lba:
                raise ValueError(
                    f"Invalid {source} GPT partition {part_num + 1} range "
                    f"[{partition.first_lba}, {partition.last_lba}] outside usable range "
                    f"[{header.first_usable_lba}, {header.last_usable_lba}]"
                )

            entries.append(partition)
        return entries

    @staticmethod
    def _used_entries(copy: GPTCopy) -> list[tuple[bytes, bytes, int, int, int, str]]:
        return [
            (
                bytes(entry.type_guid),
                bytes(entry.partition_guid),
                entry.first_lba,
                entry.last_lba,
                entry.attribute_flags,
                entry.name.decode("utf-16-le", "ignore").split("\x00")[0].strip("\x00").strip("￿"),
            )
            for entry in copy.entries
            if entry.type_guid != b"\x00" * 16
        ]

    def _compare_copies(self) -> tuple[list[str], list[str]]:
        """Compare two trustworthy copies.

        Only differences in the partition layout itself are conflicts: they decide which
        partitions are read. Geometry differences are reported as warnings, because they
        are visible cross-checks but the partition ranges still agree.

        Returns:
            A tuple of ``(conflicts, geometry_warnings)``.
        """
        if not (self.primary.valid and self.backup.valid):
            return [], []

        conflicts = []
        warnings = []
        primary_header = self.primary.header
        backup_header = self.backup.header

        for attr in ("first_usable_lba", "last_usable_lba", "partition_table_count", "partition_entry_size"):
            primary_value = getattr(primary_header, attr)
            backup_value = getattr(backup_header, attr)
            if primary_value != backup_value:
                warnings.append(f"{attr} differs: primary={primary_value}, backup={backup_value}")

        primary_entries = self._used_entries(self.primary)
        backup_entries = self._used_entries(self.backup)
        if primary_entries != backup_entries:
            conflicts.append(f"partition entries differ: primary={primary_entries!r}, backup={backup_entries!r}")

        if primary_header.guid != backup_header.guid:
            conflicts.append(
                f"disk GUID differs: primary={UUID(bytes_le=primary_header.guid)}, "
                f"backup={UUID(bytes_le=backup_header.guid)}"
            )

        return conflicts, warnings

    def _cross_check(self) -> None:
        """Check the two copies against each other and against the disk geometry."""
        primary = self.primary
        backup = self.backup

        if (
            primary.header_valid
            and backup.header is not None
            and primary.header.backup_lba != backup.offset // self.sector_size
        ):
            primary.warnings.append(
                f"primary GPT header declares backup LBA {primary.header.backup_lba}, "
                f"but the backup copy was found at LBA {backup.offset // self.sector_size}"
            )

        if (
            backup.header_valid
            and primary.header is not None
            and backup.header.backup_lba != primary.offset // self.sector_size
        ):
            backup.warnings.append(
                f"backup GPT header points to primary LBA {backup.header.backup_lba}, "
                f"but the primary copy was found at LBA {primary.offset // self.sector_size}"
            )

        if backup.header is None and primary.header_valid:
            backup.warnings.append(f"no backup GPT header found at LBA {self.disk_last_lba}")

        if primary.header is None and backup.header_valid:
            primary.warnings.append(f"no primary GPT header found at LBA {self.primary_lba}")

    def _select(self, requested_source: str | None) -> None:
        self.conflicts: list[str] = []
        self.conflict = False
        self.geometry_warnings: list[str] = []

        valid = [copy for copy in (self.primary, self.backup) if copy.valid]
        usable = [copy for copy in (self.primary, self.backup) if copy.usable]

        if requested_source is not None:
            selected = self.copies[requested_source]
            if not selected.usable:
                raise DiskError(
                    f"Requested {requested_source} GPT copy is not usable: {selected.error or 'unknown error'}"
                )
            self.conflicts, self.geometry_warnings = self._compare_copies()
            self.conflict = bool(self.conflicts)
        elif len(valid) == 2:
            self.conflicts, self.geometry_warnings = self._compare_copies()
            if self.conflicts:
                self.conflict = True
                raise GPTConflictError(self.primary, self.backup, self.conflicts)
            selected = self.primary
        elif len(valid) == 1:
            # The other copy is damaged; a failed array checksum is not a layout conflict.
            selected = valid[0]
        elif len(usable) == 1:
            # Neither array checksum validated, but one header and entry set parsed. Use it
            # only because there is no competing layout evidence.
            selected = usable[0]
        else:
            errors = "; ".join(f"{copy.source}: {copy.error}" for copy in (self.primary, self.backup) if copy.error)
            raise DiskError(f"No valid GPT partition table found ({errors or 'primary and backup are invalid'})")

        if not selected.array_valid:
            selected.warnings.append(
                f"using {selected.source} GPT copy with a failed partition array checksum; "
                "partition layout is degraded evidence"
            )

        self.selected = selected
        self.selected_source = selected.source
        self.source = selected.source
        self.other = self.backup if selected is self.primary else self.primary
        self.other_source = self.other.source
        self.other_status = self.other.status
        self.primary_status = self.primary.status
        self.backup_status = self.backup.status
        self.recovered = selected.source == "backup"
        self.degraded = not selected.valid
        self.header = selected.header
        self.gpt = selected.header
        self.warnings: list[str] = list(self.geometry_warnings)
        for copy in (self.primary, self.backup):
            self.warnings.extend(copy.warnings)

    @property
    def status(self) -> dict[str, object]:
        """Validation status of the selected and rejected copies."""
        return {
            "selected": self.selected_source,
            "primary": self.primary_status,
            "backup": self.backup_status,
            "recovered": self.recovered,
            "degraded": self.degraded,
            "conflict": self.conflict,
        }

    def as_dict(self) -> dict[str, object]:
        """Return a serializable summary of the selection and both copies."""
        return {
            "selected_source": self.selected_source,
            "recovered": self.recovered,
            "degraded": self.degraded,
            "conflict": self.conflict,
            "conflicts": list(self.conflicts),
            "geometry_warnings": list(self.geometry_warnings),
            "warnings": list(self.warnings),
            "primary": self.primary.as_dict(),
            "backup": self.backup.as_dict(),
        }

    def _partitions(self) -> Iterator[Partition]:
        # First iterate MBR partitions
        # When we find the GPT partition, iterate GPT partitions
        # This should support Hybrid GPT
        seen_gpt = False

        for mbr_part in self.mbr.partitions:
            # There can be multiple protective MBR entries
            if mbr_part.type == 0xEE:
                if seen_gpt:
                    continue

                # Numbering a hybrid GPT is a bit weird, I think it's best to do nothing
                # This way the MBR and the GPT have their own separate partition numbers
                for part_num, partition in enumerate(self.selected.entries):
                    if partition.first_lba == 0:
                        continue

                    name = (
                        partition.name.decode("utf-16-le", "ignore")
                        .split("\x00")[0]
                        .strip("\x00")
                        .strip("￿")  # a non-character in UTF-16
                    )

                    yield Partition(
                        disk=self,
                        number=part_num + 1,  # partitions are 1-indexed
                        offset=partition.first_lba * self.sector_size,
                        # last_lba is inclusive, so include the final sector
                        size=(partition.last_lba - partition.first_lba + 1) * self.sector_size,
                        type=UUID(bytes_le=partition.type_guid),
                        name=name,
                        flags=partition.attribute_flags,
                        guid=UUID(bytes_le=partition.partition_guid),
                        raw=partition,
                    )

                seen_gpt = True
            else:
                yield mbr_part
