from __future__ import annotations

import io

import pytest

from dissect.util.stream import RangeStream
from dissect.volume.disk.schemes.bsd import BSD, c_bsd
from dissect.volume.exceptions import DiskError

# Slice-relative layout shared by every synthetic 32 bit label:
#   slot 0 -> relative sector 4, fill 0x11
#   slot 1 -> relative sector 6, fill 0x22
#   slot 4 -> relative sector 8, fill 0x44
PLAN32 = {0: (4, 1, 7, 0x11), 1: (6, 1, 1, 0x22), 4: (8, 1, 27, 0x44)}
EXPECTED32 = {
    1: (4, 0x11),
    2: (6, 0x22),
    5: (8, 0x44),
}


def build_bsd32(sector_size: int, raw_base_sector: int, raw_size: int = 100, sectors: int = 32) -> bytes:
    """Build a 32 bit disklabel image.

    ``raw_base_sector`` is the whole-disk sector stored in the raw (slot 2) entry. For a standalone
    disk this is 0; for a label nested in a parent slice this is the slice's whole-disk start LBA.
    """
    entries = b""
    for slot in range(c_bsd.BSD_NPARTS_MIN):
        entry = c_bsd.partition()
        if slot == c_bsd.BSD_PART_RAW:
            entry.p_size = raw_size
            entry.p_offset = raw_base_sector
        elif slot in PLAN32:
            rel_sector, size, fstype, _ = PLAN32[slot]
            entry.p_size = size
            entry.p_offset = raw_base_sector + rel_sector
            entry.p_fstype = fstype
        entries += entry.dumps()

    label = c_bsd.disklabel(
        d_magic=c_bsd.BSD_MAGIC,
        d_magic2=c_bsd.BSD_MAGIC,
        d_secsize=sector_size,
        d_npartitions=c_bsd.BSD_NPARTS_MIN,
        d_bbsize=8192,
    )
    head = label.dumps()

    image = bytearray(sectors * sector_size)
    image[sector_size : sector_size + len(head)] = head
    image[sector_size + len(head) : sector_size + len(head) + len(entries)] = entries
    for slot, (rel_sector, size, _fstype, fill) in PLAN32.items():
        start = rel_sector * sector_size
        image[start : start + size * sector_size] = bytes([fill]) * (size * sector_size)
    return bytes(image)


def embed(parent: bytes, parent_lba: int, sector_size: int) -> bytes:
    disk = bytearray(b"\xCC" * ((parent_lba * sector_size) + len(parent)))
    disk[parent_lba * sector_size : parent_lba * sector_size + len(parent)] = parent
    return bytes(disk)


def assert_32_matches(vs: BSD, sector_size: int) -> None:
    assert vs.type == 32
    by_number = {p.number: p for p in vs.partitions}
    assert set(by_number) == set(EXPECTED32)
    for number, (rel_sector, fill) in EXPECTED32.items():
        partition = by_number[number]
        assert partition.offset == rel_sector * sector_size
        assert partition.size == sector_size
        assert partition.open().read(sector_size) == bytes([fill]) * sector_size


@pytest.mark.parametrize("sector_size", [512, 4096])
def test_bsd32_whole_disk_and_parent_view_are_identical(sector_size: int) -> None:
    parent_lba = 200

    standalone = build_bsd32(sector_size, raw_base_sector=0)
    nested_slice = build_bsd32(sector_size, raw_base_sector=parent_lba)
    disk_image = embed(nested_slice, parent_lba, sector_size)

    whole = BSD(io.BytesIO(standalone), sector_size=sector_size)
    parent = BSD(
        RangeStream(io.BytesIO(disk_image), parent_lba * sector_size, len(nested_slice)),
        sector_size=sector_size,
    )

    # Same logical label, two views: every slice-relative offset and the bytes behind it must match.
    assert [(p.number, p.offset, p.size) for p in whole.partitions] == [
        (p.number, p.offset, p.size) for p in parent.partitions
    ]
    assert_32_matches(whole, sector_size)
    assert_32_matches(parent, sector_size)


def test_bsd32_does_not_use_current_file_position() -> None:
    sector_size = 512
    parent_lba = 200
    nested_slice = build_bsd32(sector_size, raw_base_sector=parent_lba)
    disk_image = embed(nested_slice, parent_lba, sector_size)

    # A zero-based RangeStream is the supported way to hand over a parent partition.
    good = BSD(RangeStream(io.BytesIO(disk_image), parent_lba * sector_size, len(nested_slice)))
    assert_32_matches(good, sector_size)

    # A raw handle that is merely seek()'d to the parent start cannot express a zero-based view.
    # The parser must anchor to absolute 0 and therefore reject the handle instead of reading
    # filler bytes through double-counted coordinates.
    pre_seeked = io.BytesIO(disk_image)
    pre_seeked.seek(parent_lba * sector_size)
    with pytest.raises(DiskError, match="Invalid BSD disklabel magic"):
        BSD(pre_seeked)


def test_bsd32_standalone_empty_raw_slot_is_accepted() -> None:
    # A standalone disk legitimately leaves the raw slot empty (matches the shipped bsd.bin).
    sector_size = 512
    standalone = build_bsd32(sector_size, raw_base_sector=0, raw_size=0)
    vs = BSD(io.BytesIO(standalone), sector_size=sector_size)
    assert_32_matches(vs, sector_size)


def test_bsd32_nested_empty_raw_slot_with_absolute_offsets_is_rejected() -> None:
    sector_size = 512
    parent_lba = 200

    # Raw slot is empty, yet data entries keep absolute whole-disk sector numbers (>= parent LBA).
    entries = b""
    for slot in range(c_bsd.BSD_NPARTS_MIN):
        entry = c_bsd.partition()
        if slot in PLAN32:
            rel_sector, size, fstype, _ = PLAN32[slot]
            entry.p_offset = parent_lba + rel_sector
            entry.p_size = size
            entry.p_fstype = fstype
        entries += entry.dumps()

    label = c_bsd.disklabel(
        d_magic=c_bsd.BSD_MAGIC,
        d_magic2=c_bsd.BSD_MAGIC,
        d_secsize=sector_size,
        d_npartitions=c_bsd.BSD_NPARTS_MIN,
        d_bbsize=8192,
    )
    head = label.dumps()
    parent = bytearray(16 * sector_size)
    parent[sector_size : sector_size + len(head)] = head
    parent[sector_size + len(head) : sector_size + len(head) + len(entries)] = entries

    disk_image = embed(bytes(parent), parent_lba, sector_size)
    handle = RangeStream(io.BytesIO(disk_image), parent_lba * sector_size, len(parent))
    with pytest.raises(DiskError, match="raw partition .* is empty"):
        BSD(handle)


def build_bsd64() -> bytes:
    sector_size = 512
    plan = {
        0: (0x100000, 0x1000, 7, 0xA1),
        1: (0x101000, 0x1000, 7, 0xA2),
        4: (0x102000, 0x1000, 7, 0xA4),
    }
    entries = b""
    for slot in range(c_bsd.MAXPARTITIONS64):
        entry = c_bsd.partition64()
        if slot in plan:
            boffset, bsize, fstype, _fill = plan[slot]
            entry.p_boffset = boffset
            entry.p_bsize = bsize
            entry.p_fstype = fstype
        entries += entry.dumps()

    label = c_bsd.disklabel64(d_magic=c_bsd.DISKMAGIC64, d_npartitions=c_bsd.MAXPARTITIONS64)
    head = label.dumps()
    image = bytearray(0x103000)
    image[sector_size : sector_size + len(head)] = head
    image[sector_size + len(head) : sector_size + len(head) + len(entries)] = entries
    for _slot, (boffset, bsize, _fstype, fill) in plan.items():
        image[boffset : boffset + bsize] = bytes([fill]) * bsize
    return bytes(image)


def test_bsd64_offsets_are_slice_relative_bytes() -> None:
    parent = build_bsd64()
    parent_base = 0x200000
    disk_image = embed(parent, parent_base // 512, 512)

    vs = BSD(RangeStream(io.BytesIO(disk_image), parent_base, len(parent)))
    assert vs.type == 64
    expected = {1: (0x100000, 0xA1), 2: (0x101000, 0xA2), 5: (0x102000, 0xA4)}
    by_number = {p.number: p for p in vs.partitions}
    assert set(by_number) == set(expected)
    for number, (offset, fill) in expected.items():
        partition = by_number[number]
        assert partition.offset == offset
        assert partition.open().read(4) == bytes([fill]) * 4


def test_bsd64_does_not_use_current_file_position() -> None:
    parent = build_bsd64()
    parent_base = 0x200000
    disk_image = embed(parent, parent_base // 512, 512)

    handle = io.BytesIO(disk_image)
    handle.seek(parent_base)
    with pytest.raises(DiskError, match="Invalid BSD disklabel magic"):
        BSD(handle)
