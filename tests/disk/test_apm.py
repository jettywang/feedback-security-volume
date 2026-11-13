from __future__ import annotations

import io
import struct
from typing import BinaryIO

import pytest

from dissect.volume import disk
from dissect.volume.disk.schemes import APM
from dissect.volume.exceptions import DiskError

# A byte value that is never valid inside a partition entry or its text fields, so any parser that
# wanders into the padding reads conspicuous garbage instead of zeroes.
APM_PADDING = b"\xaa"

# (name, type, size in logical blocks)
APM_SPECS = [
    ("Apple", "Apple_partition_map", 1),
    ("FirstData", "Windows_NTFS", 2),
    ("FreeSpace", "Apple_Free", 1),
]
APM_PAYLOAD = b"NTFS-PAYLOAD-MARKER!!"


def _partition_entry(count: int, start: int, size: int, name: str, ptype: str) -> bytes:
    entry = bytearray(512)
    entry[0:2] = b"PM"
    struct.pack_into(">I", entry, 4, count)  # partition_count
    struct.pack_into(">I", entry, 8, start)  # partition_start, in logical blocks
    struct.pack_into(">I", entry, 12, size)  # partition_size, in logical blocks
    entry[16:48] = name.encode().ljust(32, b"\x00")
    entry[48:80] = ptype.encode().ljust(32, b"\x00")
    return bytes(entry)


def _build_apm(block_size: int, with_ddr: bool = True) -> tuple[bytes, int, int]:
    """Build an APM image whose blocks are filled with conspicuous padding.

    Returns the image, the total number of blocks and the block index of the partition map.
    """
    partition_count = len(APM_SPECS)
    map_start = 1 if with_ddr else 0
    total_blocks = map_start + partition_count + sum(spec[2] for spec in APM_SPECS)
    image = bytearray(APM_PADDING * (total_blocks * block_size))

    if with_ddr:
        ddr = bytearray(block_size)
        ddr[0:2] = b"ER"
        struct.pack_into(">H", ddr, 2, block_size)  # Driver Descriptor Record block size
        struct.pack_into(">I", ddr, 4, total_blocks)
        image[0:block_size] = ddr

    for i, (name, ptype, part_size) in enumerate(APM_SPECS):
        start = map_start + partition_count + sum(spec[2] for spec in APM_SPECS[:i])
        entry_offset = (map_start + i) * block_size
        image[entry_offset : entry_offset + 512] = _partition_entry(partition_count, start, part_size, name, ptype)
        if name == "FirstData":
            data_offset = start * block_size
            image[data_offset : data_offset + len(APM_PAYLOAD)] = APM_PAYLOAD

    return bytes(image), total_blocks, map_start


def test_apm(apm: BinaryIO) -> None:
    d = disk.Disk(apm)

    assert isinstance(d.scheme, APM)
    assert len(d.partitions) == 3
    assert d.scheme.apm.signature == b"PM"

    assert d.partitions[0].number == 1
    assert d.partitions[0].offset == 0x200
    assert d.partitions[0].size == 0x7E00
    assert d.partitions[0].type == "Apple_partition_map"
    assert d.partitions[0].name == "Apple"
    assert repr(d.partitions[0]) == (
        "<Partition number=1 offset=0x200 size=0x7e00 type='Apple_partition_map' name='Apple'>"
    )

    assert d.partitions[1].number == 2
    assert d.partitions[1].offset == 0x8000
    assert d.partitions[1].size == 0x752FF6000
    assert d.partitions[1].type == "Windows_NTFS"
    assert d.partitions[1].name == ""

    assert d.partitions[2].number == 3
    assert d.partitions[2].offset == 0x752FFE000
    assert d.partitions[2].size == 0x2000
    assert d.partitions[2].type == "Apple_Free"
    assert d.partitions[2].name == ""


def test_apm_invalid() -> None:
    buf = io.BytesIO(512 * b"\x00")
    with pytest.raises(disk.DiskError):
        APM(buf)


@pytest.mark.parametrize("block_size", [512, 2048])
def test_apm_non_512_blocks(block_size: int) -> None:
    image, total_blocks, map_start = _build_apm(block_size)
    apm = APM(io.BytesIO(image), sector_size=block_size)

    assert apm.sector_size == block_size
    assert apm.ddr is not None
    assert apm.ddr.signature == b"ER"
    assert apm.ddr.block_size == block_size
    assert apm.ddr.block_count == total_blocks
    assert apm.apm.signature == b"PM"

    assert len(apm.partitions) == len(APM_SPECS)

    data_start_block = map_start + len(APM_SPECS)
    expected_start = data_start_block
    for i, (partition, (name, ptype, part_size)) in enumerate(zip(apm.partitions, APM_SPECS)):
        # Each raw entry lives in its own logical block, never at a back-to-back 512-byte stride.
        entry_offset = (map_start + i) * block_size
        assert image[entry_offset : entry_offset + 2] == b"PM"
        assert partition.raw.signature == b"PM"

        # On blocks larger than the fixed 512-byte entry, the remainder is padding.
        if block_size > 512:
            assert image[entry_offset + 512 : entry_offset + block_size] == APM_PADDING * (block_size - 512)

        # partition_start / partition_size are logical blocks and must be converted to bytes.
        assert partition.number == i + 1
        assert partition.name == name
        assert partition.type == ptype
        assert partition.offset == expected_start * block_size
        assert partition.size == part_size * block_size

        # Opening the partition reads its own bytes and never the inter-entry padding.
        with partition.open() as fh:
            assert fh.read() == image[partition.offset : partition.offset + partition.size]

        expected_start += part_size

    # The data partition exposes the payload followed by its block padding.
    data_partition = apm.partitions[1]
    with data_partition.open() as fh:
        assert fh.read(len(APM_PAYLOAD)) == APM_PAYLOAD
        assert fh.read() == APM_PADDING * (data_partition.size - len(APM_PAYLOAD))


@pytest.mark.parametrize("block_size", [512, 2048])
def test_apm_disk_detects_block_size(block_size: int) -> None:
    image, _, _ = _build_apm(block_size)

    detected = disk.Disk(io.BytesIO(image))
    assert isinstance(detected.scheme, APM)
    assert detected.sector_size == block_size
    assert [partition.name for partition in detected.partitions] == [spec[0] for spec in APM_SPECS]

    # Providing the matching block size explicitly produces identical coordinates.
    explicit = disk.Disk(io.BytesIO(image), sector_size=block_size)
    assert explicit.sector_size == block_size
    assert [(p.offset, p.size) for p in explicit.partitions] == [(p.offset, p.size) for p in detected.partitions]


def test_apm_sector_size_mismatch() -> None:
    image, _, _ = _build_apm(2048)

    with pytest.raises(DiskError, match="Sector size mismatch"):
        APM(io.BytesIO(image), sector_size=512)

    with pytest.raises(disk.DiskError):
        disk.Disk(io.BytesIO(image), sector_size=512)


@pytest.mark.parametrize("block_size", [512, 2048])
def test_apm_without_driver_descriptor_record(block_size: int) -> None:
    image, _, map_start = _build_apm(block_size, with_ddr=False)

    # No ER block: the caller-provided block size keeps a well-defined meaning and entries are still
    # spaced one logical block apart.
    apm = APM(io.BytesIO(image), sector_size=block_size)
    assert apm.ddr is None
    assert apm.sector_size == block_size
    assert [partition.name for partition in apm.partitions] == [spec[0] for spec in APM_SPECS]
    assert apm.partitions[0].offset == (map_start + len(APM_SPECS)) * block_size

    # Without an ER block and without an explicit size, the 512-byte default applies.
    default_apm = APM(io.BytesIO(_build_apm(512, with_ddr=False)[0]))
    assert default_apm.ddr is None
    assert default_apm.sector_size == 512
    assert len(default_apm.partitions) == len(APM_SPECS)


@pytest.mark.parametrize("block_size", [512, 2048])
def test_apm_truncated_entries(block_size: int) -> None:
    image, _, map_start = _build_apm(block_size)

    # Truncated first entry: not enough bytes for a full structure.
    truncated_first = io.BytesIO(image[: map_start * block_size + 100])
    with pytest.raises((DiskError, EOFError)):
        APM(truncated_first, sector_size=block_size)

    # Only the first entry block present: later entries run into EOF instead of yielding garbage.
    truncated_later = io.BytesIO(image[: (map_start + 1) * block_size])
    with pytest.raises((DiskError, EOFError)):
        APM(truncated_later, sector_size=block_size)


def test_apm_non_pm_entry_rejected() -> None:
    image, _, map_start = _build_apm(2048)
    broken = bytearray(image)
    # Corrupt the second entry's signature so a per-block stride lands on a non-PM structure.
    broken[(map_start + 1) * 2048 : (map_start + 1) * 2048 + 2] = APM_PADDING * 2

    with pytest.raises(DiskError, match="Invalid APM partition entry 2"):
        APM(io.BytesIO(bytes(broken)), sector_size=2048)
