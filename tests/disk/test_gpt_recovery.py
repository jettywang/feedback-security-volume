from __future__ import annotations

import io
import struct
import zlib
from uuid import UUID

import pytest

from dissect.volume import disk
from dissect.volume.disk.schemes import GPT
from dissect.volume.exceptions import GPTConflictError

GPT_TYPE = UUID("0fc63daf-8483-4772-8e79-3d69d8477de4")

HEADER_FORMAT = "<8sIIIIQQQQ16sQIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


ENTRY_COUNT = 128
ENTRY_SIZE = 128


def geometry(sector_size: int, sector_count: int) -> tuple[int, int, int, int]:
    """Return ``(array_sectors, primary_array_lba, first_usable, last_usable)``."""
    array_sectors = (ENTRY_COUNT * ENTRY_SIZE + sector_size - 1) // sector_size
    primary_array_lba = 2
    first_usable = primary_array_lba + array_sectors
    last_usable = sector_count - 1 - array_sectors - 1
    return array_sectors, primary_array_lba, first_usable, last_usable


def build_disk(
    sector_size: int,
    sector_count: int,
    partitions: list[tuple[int, int]],
    *,
    primary_partitions: list[tuple[int, int]] | None = None,
    corrupt_primary_header: bool = False,
    corrupt_primary_array: bool = False,
    corrupt_backup_array: bool = False,
) -> bytearray:
    """Build a small GPT disk with optional damage.

    Partition ranges are inclusive ``(first_lba, last_lba)`` pairs. Data sectors are
    filled with a per-sector pattern so the first and last byte of every partition can
    be verified against its actual extent.
    """
    disk_data = bytearray(sector_count * sector_size)

    array_sectors, primary_array_lba, first_usable, last_usable = geometry(sector_size, sector_count)
    backup_header_lba = sector_count - 1
    backup_array_lba = backup_header_lba - array_sectors
    entry_count = ENTRY_COUNT

    # Protective MBR
    mbr = bytearray(sector_size)
    mbr[0x1BE : 0x1BE + 16] = struct.pack("<BBHBBHII", 0x00, 0x00, 0x0200, 0xEE, 0xFF, 0xFFFF, 1, sector_count - 1)
    mbr[510:512] = b"\x55\xaa"
    disk_data[:sector_size] = mbr

    def build_array(ranges: list[tuple[int, int]]) -> bytes:
        array = bytearray(entry_count * 128)
        for index, (first_lba, last_lba) in enumerate(ranges):
            entry = struct.pack(
                "<16s16sQQQ",
                GPT_TYPE.bytes_le,
                UUID(int=index + 1).bytes_le,
                first_lba,
                last_lba,
                0,
            )
            array[index * 128 : index * 128 + len(entry)] = entry
        return bytes(array)

    def build_header(
        current_lba: int,
        alternate_lba: int,
        array_lba: int,
        array: bytes,
        guid: bytes,
    ) -> bytes:
        header = bytearray(sector_size)
        struct.pack_into(
            HEADER_FORMAT,
            header,
            0,
            b"EFI PART",
            0x10000,
            HEADER_SIZE,
            0,
            0,
            current_lba,
            alternate_lba,
            first_usable,
            last_usable,
            guid,
            array_lba,
            entry_count,
            128,
            zlib.crc32(array) & 0xFFFFFFFF,
        )
        header[16:20] = b"\x00\x00\x00\x00"
        struct.pack_into("<I", header, 16, zlib.crc32(bytes(header[:HEADER_SIZE])) & 0xFFFFFFFF)
        return bytes(header)

    primary_ranges = primary_partitions if primary_partitions is not None else partitions
    backup_ranges = partitions
    guid = UUID("12345678-1234-5678-9abc-def012345678").bytes_le

    primary_array = build_array(primary_ranges)
    backup_array = build_array(backup_ranges)

    # Checksums are computed before any damage, so a corrupted array no longer matches
    # the checksum recorded in its header.
    primary_header = build_header(1, backup_header_lba, primary_array_lba, primary_array, guid)
    backup_header = build_header(backup_header_lba, 1, backup_array_lba, backup_array, guid)

    damaged_primary_array = bytearray(primary_array)
    if corrupt_primary_array:
        damaged_primary_array[0] = (damaged_primary_array[0] + 0x17) & 0xFF
    primary_array = bytes(damaged_primary_array)

    if corrupt_backup_array:
        backup_array = bytearray(backup_array)
        backup_array[0] = (backup_array[0] + 0x17) & 0xFF
        backup_array = bytes(backup_array)

    if corrupt_primary_header:
        primary_header = bytearray(primary_header)
        primary_header[16] ^= 0xFF  # break the header checksum
        primary_header = bytes(primary_header)

    primary_array_begin = primary_array_lba * sector_size
    primary_array_end = (primary_array_lba + array_sectors) * sector_size
    backup_array_begin = backup_array_lba * sector_size
    backup_array_end = (backup_array_lba + array_sectors) * sector_size
    disk_data[primary_array_begin:primary_array_end] = bytes(primary_array)
    disk_data[backup_array_begin:backup_array_end] = backup_array

    disk_data[sector_size : 2 * sector_size] = primary_header
    disk_data[backup_header_lba * sector_size :] = backup_header

    for partition_id, (first_lba, last_lba) in enumerate(partitions, start=1):
        for lba in range(first_lba, last_lba + 1):
            fill = (partition_id * 17 + lba) & 0xFF
            disk_data[lba * sector_size : (lba + 1) * sector_size] = bytes([fill]) * sector_size

    return disk_data


def assert_readable(part, image: bytes, sector_size: int) -> None:
    """Compare the partition stream boundaries with the actual image bytes."""
    first_byte = image[part.offset]
    last_byte = image[part.offset + part.size - 1]

    stream = part.open()
    assert stream.read(1) == bytes([first_byte])

    stream.seek(part.size - 1)
    assert stream.read(1) == bytes([last_byte])
    assert stream.read(1) == b""

    # The final LBA is inclusive, so the partition ends exactly on a sector boundary and
    # includes the whole last sector.
    assert part.offset % sector_size == 0
    assert part.size == (part.raw.last_lba - part.raw.first_lba + 1) * sector_size
    assert (part.offset + part.size) % sector_size == 0

    # Both boundary bytes are the pattern of the sector they belong to.
    assert first_byte == (part.number * 17 + part.raw.first_lba) & 0xFF
    assert last_byte == (part.number * 17 + part.raw.last_lba) & 0xFF


def ranges_for(sector_size: int, sector_count: int) -> list[tuple[int, int]]:
    _, _, first_usable, last_usable = geometry(sector_size, sector_count)
    third = (last_usable - first_usable) // 3
    return [
        (first_usable + 1, first_usable + third),
        (first_usable + third + 2, last_usable),
    ]


@pytest.mark.parametrize("sector_size", [512, 4096])
def test_primary_header_damaged(sector_size: int) -> None:
    sector_count = 240 if sector_size == 512 else 64
    partitions = ranges_for(sector_size, sector_count)
    data = build_disk(sector_size, sector_count, partitions, corrupt_primary_header=True)
    original = bytes(data)

    d = disk.Disk(io.BytesIO(data))
    assert isinstance(d.scheme, GPT)
    assert d.scheme.selected_source == "backup"
    assert d.scheme.recovered is True
    assert d.scheme.primary.header_valid is False
    assert d.scheme.primary.valid is False
    assert d.scheme.primary.error is not None
    assert d.scheme.backup.valid is True
    assert len(d.partitions) == 2

    for part in d.partitions:
        assert_readable(part, original, sector_size)

    # Reading must not modify the source image.
    assert bytes(data) == original


@pytest.mark.parametrize("sector_size", [512, 4096])
def test_primary_array_damaged(sector_size: int) -> None:
    sector_count = 240 if sector_size == 512 else 64
    partitions = ranges_for(sector_size, sector_count)
    data = build_disk(sector_size, sector_count, partitions, corrupt_primary_array=True)

    d = disk.Disk(io.BytesIO(data))
    assert isinstance(d.scheme, GPT)
    assert d.scheme.selected_source == "backup"
    assert d.scheme.recovered is True
    assert d.scheme.primary.header_valid is True
    assert d.scheme.primary.array_valid is False
    assert "partition array checksum" in d.scheme.primary.array_error
    assert d.scheme.backup.valid is True
    assert d.scheme.status["primary"] == "invalid"
    assert d.scheme.status["backup"] == "valid"

    for part in d.partitions:
        assert_readable(part, bytes(data), sector_size)


@pytest.mark.parametrize("sector_size", [512, 4096])
def test_backup_array_damaged_uses_primary(sector_size: int) -> None:
    sector_count = 240 if sector_size == 512 else 64
    partitions = ranges_for(sector_size, sector_count)
    data = build_disk(sector_size, sector_count, partitions, corrupt_backup_array=True)
    original = bytes(data)

    d = disk.Disk(io.BytesIO(data))
    assert d.scheme.selected_source == "primary"
    assert d.scheme.recovered is False
    assert d.scheme.backup.header_valid is True
    assert d.scheme.backup.array_valid is False
    assert d.scheme.primary.valid is True
    assert d.scheme.other_status == "invalid"
    assert "backup" in d.scheme.backup.array_error

    for part in d.partitions:
        assert_readable(part, original, sector_size)


@pytest.mark.parametrize("sector_size", [512, 4096])
def test_conflicting_copies_require_source(sector_size: int) -> None:
    sector_count = 240 if sector_size == 512 else 64
    _, _, first_usable, last_usable = geometry(sector_size, sector_count)
    backup_partitions = ranges_for(sector_size, sector_count)
    primary_partitions = [(backup_partitions[0][0], backup_partitions[0][0] + 10), backup_partitions[1]]
    data = build_disk(
        sector_size,
        sector_count,
        partitions=backup_partitions,
        primary_partitions=primary_partitions,
    )

    with pytest.raises(GPTConflictError) as exc:
        disk.Disk(io.BytesIO(data))

    assert exc.value.primary.valid is True
    assert exc.value.backup.valid is True
    assert any("partition entries differ" in conflict for conflict in exc.value.conflicts)

    # No auto-repair and no silent choice: the image is unchanged.
    assert bytes(data) == build_disk(
        sector_size,
        sector_count,
        partitions=backup_partitions,
        primary_partitions=primary_partitions,
    )

    gpt_primary = GPT(io.BytesIO(data), sector_size=sector_size, source="primary")
    assert gpt_primary.selected_source == "primary"
    assert gpt_primary.conflict is True
    assert gpt_primary.other.source == "backup"
    assert gpt_primary.other_status == "valid"
    assert [(first, last) for _, first, last in gpt_primary.selected.partition_ranges] == primary_partitions

    gpt_backup = GPT(io.BytesIO(data), sector_size=sector_size, source="backup")
    assert gpt_backup.selected_source == "backup"
    assert gpt_backup.selected.partition_ranges and gpt_backup.other_status == "valid"
    assert [(first, last) for _, first, last in gpt_backup.selected.partition_ranges] == backup_partitions

    selected = disk.Disk(io.BytesIO(data), gpt_source="primary")
    assert selected.scheme.selected_source == "primary"
    assert [part.size for part in selected.partitions] == [
        (last - first + 1) * sector_size for first, last in primary_partitions
    ]


@pytest.mark.parametrize("sector_size", [512, 4096])
def test_geometry_difference_is_reported_not_conflict(sector_size: int) -> None:
    sector_count = 240 if sector_size == 512 else 64
    _, _, first_usable, last_usable = geometry(sector_size, sector_count)
    partitions = [(first_usable + 1, last_usable - 2)]
    data = build_disk(sector_size, sector_count, partitions)

    # Only the backup header's last usable LBA differs; the partition entries stay valid
    # and identical. This is a geometry disagreement, not a layout conflict.
    backup_lba = sector_count - 1
    offset = backup_lba * sector_size
    header = bytearray(data[offset : offset + sector_size])
    struct.pack_into("<Q", header, 48, last_usable - 1)
    header[16:20] = b"\x00\x00\x00\x00"
    struct.pack_into("<I", header, 16, zlib.crc32(bytes(header[:HEADER_SIZE])) & 0xFFFFFFFF)
    data[offset : offset + sector_size] = header

    d = disk.Disk(io.BytesIO(data))
    assert d.scheme.selected_source == "primary"
    assert d.scheme.conflict is False
    assert d.scheme.primary.valid is True
    assert d.scheme.backup.valid is True
    assert any("last_usable_lba differs" in warning for warning in d.scheme.geometry_warnings)


@pytest.mark.parametrize("sector_size", [512, 4096])
def test_both_copies_bad(sector_size: int) -> None:
    sector_count = 240 if sector_size == 512 else 64
    _, _, first_usable, last_usable = geometry(sector_size, sector_count)
    data = build_disk(sector_size, sector_count, [(first_usable + 1, last_usable)], corrupt_primary_header=True)

    # Damage the backup header as well.
    backup_header_lba = sector_count - 1
    data[backup_header_lba * sector_size] ^= 0xFF

    with pytest.raises(disk.DiskError, match="No valid GPT partition table found"):
        disk.Disk(io.BytesIO(data))


@pytest.mark.parametrize("sector_size", [512, 4096])
def test_array_boundary_last_sector_included(sector_size: int) -> None:
    sector_count = 240 if sector_size == 512 else 64
    _, _, first_usable, last_usable = geometry(sector_size, sector_count)
    partitions = [(first_usable + 1, last_usable)]
    data = build_disk(sector_size, sector_count, partitions)

    d = disk.Disk(io.BytesIO(data))
    part = d.partitions[0]
    assert part.raw.last_lba == last_usable
    assert part.size == (last_usable - part.raw.first_lba + 1) * sector_size

    assert_readable(part, bytes(data), sector_size)

    # The last data sector is followed by the backup partition array. Accessing the byte
    # after the partition must not be possible through the partition stream, but must be
    # present in the image: this distinguishes an inclusive last LBA from a range that
    # silently stops one sector short.
    boundary = part.offset + part.size
    assert boundary == (last_usable + 1) * sector_size
    assert part.raw.last_lba == (boundary // sector_size) - 1
    assert len(data) > boundary
