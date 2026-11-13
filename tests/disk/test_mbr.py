from __future__ import annotations

import io
import struct
from typing import BinaryIO

import pytest

from dissect.volume import disk
from dissect.volume.disk.schemes import MBR

SECTOR_SIZE = 512


def _partition_entry(type_: int, sector_ofs: int, sector_size: int, bootable: int = 0) -> bytes:
    entry = bytearray(16)
    entry[0] = bootable
    entry[4] = type_
    struct.pack_into("<II", entry, 8, sector_ofs, sector_size)
    return bytes(entry)


def _mbr_sector(*entries: bytes) -> bytes:
    buf = bytearray(SECTOR_SIZE)
    for idx, entry in enumerate(entries):
        buf[0x1BE + (idx * 16) : 0x1BE + ((idx + 1) * 16)] = entry
    struct.pack_into("<H", buf, 0x1FE, 0xAA55)
    return bytes(buf)


def _disk_image(writes: dict[int, bytes], start: int = 0, sectors: int = 1024) -> BinaryIO:
    """Build an in-memory disk image with an optional non-zero start of the disk stream."""
    buf = bytearray(start + (sectors * SECTOR_SIZE))
    for lba, data in writes.items():
        offset = start + (lba * SECTOR_SIZE)
        buf[offset : offset + len(data)] = data

    fh = io.BytesIO(bytes(buf))
    fh.seek(start)
    return fh


def _partitions(fh: BinaryIO) -> list[tuple[int, int, int, int]]:
    return [(p.number, p.offset, p.size, p.type) for p in MBR(fh).partitions]


def test_mbr(mbr: BinaryIO) -> None:
    d = disk.Disk(mbr)

    assert isinstance(d.scheme, MBR)
    assert len(d.partitions) == 2
    assert d.scheme.mbr.bootsig == 0xAA55

    assert d.partitions[0].number == 1
    assert d.partitions[0].offset == 0x100000
    assert d.partitions[0].size == 0x1F400000
    assert d.partitions[0].type == 0x7
    assert repr(d.partitions[0]) == "<Partition number=1 offset=0x100000 size=0x1f400000 type='0x7 (NTFS)' name=None>"

    assert d.partitions[1].number == 2
    assert d.partitions[1].offset == 0x1F500000
    assert d.partitions[1].size == 0xEE0A00000
    assert d.partitions[1].type == 0x7


def test_mbr_invalid() -> None:
    buf = io.BytesIO(b"\x00" * 512)
    with pytest.raises(disk.DiskError):
        MBR(buf)


def test_mbr_extended_two_chains_nonzero_start() -> None:
    start = 0x1000

    image = _disk_image(
        {
            0: _mbr_sector(
                _partition_entry(0x0F, 100, 300),
                _partition_entry(0x05, 500, 200),
                _partition_entry(0x07, 800, 50),
            ),
            # First EBR of the first chain: logical partition relative to this EBR,
            # next EBR relative to the first EBR of the chain.
            100: _mbr_sector(_partition_entry(0x07, 1, 10), _partition_entry(0x0F, 20, 100)),
            120: _mbr_sector(_partition_entry(0x07, 2, 5), _partition_entry(0x0F, 40, 100)),
            140: _mbr_sector(_partition_entry(0x07, 3, 5)),
            # Second chain is a sibling extended partition and starts its own chain.
            500: _mbr_sector(_partition_entry(0x07, 1, 10), _partition_entry(0x0F, 30, 100)),
            530: _mbr_sector(_partition_entry(0x07, 2, 5)),
        },
        start=start,
        sectors=900,
    )

    partitions = _partitions(image)
    assert partitions == [
        (1, start + (101 * SECTOR_SIZE), 10 * SECTOR_SIZE, 0x07),
        (2, start + (122 * SECTOR_SIZE), 5 * SECTOR_SIZE, 0x07),
        (3, start + (143 * SECTOR_SIZE), 5 * SECTOR_SIZE, 0x07),
        (4, start + (501 * SECTOR_SIZE), 10 * SECTOR_SIZE, 0x07),
        (5, start + (532 * SECTOR_SIZE), 5 * SECTOR_SIZE, 0x07),
        (6, start + (800 * SECTOR_SIZE), 50 * SECTOR_SIZE, 0x07),
    ]


def test_mbr_extended_self_loop() -> None:
    image = _disk_image(
        {
            0: _mbr_sector(_partition_entry(0x0F, 100, 100)),
            100: _mbr_sector(_partition_entry(0x07, 1, 4), _partition_entry(0x0F, 0, 100)),
        },
        sectors=300,
    )

    partitions = _partitions(image)
    assert partitions == [(1, 101 * SECTOR_SIZE, 4 * SECTOR_SIZE, 0x07)]


def test_mbr_extended_loop_between_ebrs() -> None:
    image = _disk_image(
        {
            0: _mbr_sector(_partition_entry(0x0F, 100, 200)),
            100: _mbr_sector(_partition_entry(0x07, 1, 4), _partition_entry(0x0F, 50, 100)),
            150: _mbr_sector(_partition_entry(0x07, 1, 4), _partition_entry(0x0F, 0, 100)),
        },
        sectors=400,
    )

    partitions = _partitions(image)
    assert partitions == [
        (1, 101 * SECTOR_SIZE, 4 * SECTOR_SIZE, 0x07),
        (2, 151 * SECTOR_SIZE, 4 * SECTOR_SIZE, 0x07),
    ]


def test_mbr_extended_outside_container() -> None:
    image = _disk_image(
        {
            0: _mbr_sector(_partition_entry(0x0F, 100, 50)),
            # Second logical partition ends beyond the container, so it is not surfaced.
            100: _mbr_sector(_partition_entry(0x07, 1, 4), _partition_entry(0x07, 45, 10)),
            # The link itself also points outside the container and must not be followed.
            150: _mbr_sector(_partition_entry(0x07, 1, 5)),
            160: _mbr_sector(_partition_entry(0x07, 1, 5)),
        },
        sectors=300,
    )

    partitions = _partitions(image)
    assert partitions == [(1, 101 * SECTOR_SIZE, 4 * SECTOR_SIZE, 0x07)]


def test_mbr_extended_long_chain() -> None:
    chain_length = 100
    first_ebr = 10
    entry_size = 4

    writes = {0: _mbr_sector(_partition_entry(0x0F, first_ebr, chain_length * entry_size + 10))}
    for idx in range(chain_length):
        ebr = first_ebr + (idx * entry_size)
        entries = [_partition_entry(0x07, 1, 2)]
        if idx < chain_length - 1:
            entries.append(_partition_entry(0x0F, (idx + 1) * entry_size, 10))
        writes[ebr] = _mbr_sector(*entries)

    image = _disk_image(writes, sectors=first_ebr + (chain_length * entry_size) + 10)
    partitions = _partitions(image)

    assert len(partitions) == chain_length
    assert [p[0] for p in partitions] == list(range(1, chain_length + 1))
    assert all(p[1] == (first_ebr + (idx * entry_size) + 1) * SECTOR_SIZE for idx, p in enumerate(partitions))


def test_mbr_extended_truncated_ebr() -> None:
    # A damaged chain pointing at sector 100 that lies past the end of the stream must
    # not raise, and the partitions confirmed before the broken link are kept.
    image = _disk_image(
        {
            0: _mbr_sector(_partition_entry(0x0F, 10, 300)),
            10: _mbr_sector(_partition_entry(0x07, 1, 4), _partition_entry(0x0F, 100, 10)),
        },
        sectors=20,
    )

    partitions = _partitions(image)
    assert partitions == [(1, 11 * SECTOR_SIZE, 4 * SECTOR_SIZE, 0x07)]


def test_mbr_extended_broken_chain_keeps_following_chain() -> None:
    image = _disk_image(
        {
            0: _mbr_sector(
                _partition_entry(0x0F, 10, 60),
                _partition_entry(0x0F, 200, 60),
                _partition_entry(0x07, 400, 10),
            ),
            # First chain self-loops after one confirmed partition.
            10: _mbr_sector(_partition_entry(0x07, 1, 4), _partition_entry(0x0F, 0, 10)),
            # Second chain still has to be parsed at its own container base.
            200: _mbr_sector(_partition_entry(0x07, 1, 6), _partition_entry(0x0F, 20, 10)),
            220: _mbr_sector(_partition_entry(0x07, 2, 6)),
        },
        start=0x800,
        sectors=500,
    )

    partitions = _partitions(image)
    assert partitions == [
        (1, 0x800 + (11 * SECTOR_SIZE), 4 * SECTOR_SIZE, 0x07),
        (2, 0x800 + (201 * SECTOR_SIZE), 6 * SECTOR_SIZE, 0x07),
        (3, 0x800 + (222 * SECTOR_SIZE), 6 * SECTOR_SIZE, 0x07),
        (4, 0x800 + (400 * SECTOR_SIZE), 10 * SECTOR_SIZE, 0x07),
    ]
