from __future__ import annotations

import random
from io import BytesIO
from typing import BinaryIO

import pytest

from dissect.volume.raid.raid import PhysicalDisk, VirtualDisk
from dissect.volume.raid.stream import Level, RAID0Stream

STRIPE_SIZE = 0x1000
WORD_SIZE = 8

# Member sizes for a RAID0 set with unequal disks:
# - members 0 and 2 differ in raw length but round down to the same number of stripes
# - members 1 and 3 are larger, so a second strip zone remains on members {1, 3}
#   (non-contiguous role numbers) and a third zone on member {3} alone
MEMBER_SIZES = {
    0: 4 * STRIPE_SIZE,
    1: 6 * STRIPE_SIZE,
    2: 4 * STRIPE_SIZE + 0x200,
    3: 8 * STRIPE_SIZE,
}

ZONE_BOUNDARIES = (16 * STRIPE_SIZE, 20 * STRIPE_SIZE)
VIRTUAL_SIZE = 22 * STRIPE_SIZE


def fill_member(disk_idx: int, size: int) -> bytes:
    """Fill a member with words encoding its role number and device offset."""
    words = [
        ((disk_idx << 56) | (offset // WORD_SIZE)).to_bytes(WORD_SIZE, "little")
        for offset in range(0, size - size % WORD_SIZE, WORD_SIZE)
    ]
    return b"".join(words) + b"\xff" * (size % WORD_SIZE)


def stripe_filler_model(members: dict[int, bytes], stripe_size: int) -> bytes:
    """Independently compute the expected RAID0 content with a plain round-robin stripe walk.

    Deliberately does not reuse the zone computation of the implementation under test.
    """
    rounded = {idx: len(data) - len(data) % stripe_size for idx, data in members.items()}
    expected = bytearray()
    zone_start = 0
    for zone_end_on_disk in sorted(set(rounded.values())):
        zone_members = [idx for idx in sorted(members) if rounded[idx] > zone_start]
        for device_offset in range(zone_start, zone_end_on_disk, stripe_size):
            for idx in zone_members:
                expected += members[idx][device_offset : device_offset + stripe_size]
        zone_start = zone_end_on_disk
    return bytes(expected)


def decode_word(word: bytes) -> tuple[int, int]:
    """Decode a filler word into (member role number, device offset)."""
    value = int.from_bytes(word, "little")
    return value >> 56, (value & 0x00FFFFFFFFFFFFFF) * WORD_SIZE


def assert_logical_read(fh: BinaryIO, expected: bytes, offset: int, length: int) -> None:
    fh.seek(offset)
    got = fh.read(length)
    want = expected[offset : offset + length]
    if got == want:
        return

    diff = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
    logical = offset + diff
    word_index = logical - logical % WORD_SIZE - offset
    got_member, got_device_offset = decode_word(got[word_index : word_index + WORD_SIZE])
    want_member, want_device_offset = decode_word(want[word_index : word_index + WORD_SIZE])
    pytest.fail(
        f"read at logical {offset:#x} (length {length:#x}): logical offset {logical:#x} "
        f"should map to member {want_member} @ device {want_device_offset:#x}, "
        f"but read came from member {got_member} @ device {got_device_offset:#x}"
    )


@pytest.fixture
def raid0_unequal() -> tuple[RAID0Stream, bytes]:
    members = {idx: fill_member(idx, size) for idx, size in MEMBER_SIZES.items()}
    disk_map = {idx: (0, PhysicalDisk(BytesIO(data), 0, len(data))) for idx, data in members.items()}
    vd = VirtualDisk("raid0-unequal", "raid0-unequal", VIRTUAL_SIZE, Level.RAID0, 0, STRIPE_SIZE, 4, disk_map)
    fh = vd.open()
    assert isinstance(fh, RAID0Stream)
    return fh, stripe_filler_model(members, STRIPE_SIZE)


def test_raid0_unequal_members_zone_layout(raid0_unequal: tuple[RAID0Stream, bytes]) -> None:
    fh, _ = raid0_unequal

    # The sample must actually exercise strip zone switches with non-contiguous role numbers
    assert len(fh.zones) == 3
    assert sorted(fh.zones[0].devices) == [0, 1, 2, 3]
    assert sorted(fh.zones[1].devices) == [1, 3]
    assert sorted(fh.zones[2].devices) == [3]
    assert [zone.zone_end for zone in fh.zones] == [*ZONE_BOUNDARIES, VIRTUAL_SIZE]


def test_raid0_unequal_members_small_reads(raid0_unequal: tuple[RAID0Stream, bytes]) -> None:
    fh, expected = raid0_unequal

    # Sequential small reads across the whole set, crossing every zone switch
    for offset in range(0, VIRTUAL_SIZE, 0x200):
        assert_logical_read(fh, expected, offset, 0x200)

    # Small reads straddling each zone boundary
    for boundary in ZONE_BOUNDARIES:
        for delta in (1, 0x100, STRIPE_SIZE // 2, STRIPE_SIZE - 1):
            assert_logical_read(fh, expected, boundary - delta, 2 * delta)


def test_raid0_unequal_members_large_reads(raid0_unequal: tuple[RAID0Stream, bytes]) -> None:
    fh, expected = raid0_unequal

    offsets = [0, STRIPE_SIZE - 1]
    for boundary in ZONE_BOUNDARIES:
        offsets += [boundary - 3 * STRIPE_SIZE, boundary - 1, boundary, boundary + STRIPE_SIZE - 1]
    offsets += [VIRTUAL_SIZE - 5 * STRIPE_SIZE, VIRTUAL_SIZE - 4 * STRIPE_SIZE - 1]

    for offset in offsets:
        for length in (2 * STRIPE_SIZE, 4 * STRIPE_SIZE + 0x123):
            if offset + length <= VIRTUAL_SIZE:
                assert_logical_read(fh, expected, offset, length)


def test_raid0_unequal_members_random_seek(raid0_unequal: tuple[RAID0Stream, bytes]) -> None:
    fh, expected = raid0_unequal

    rng = random.Random(0xA110)
    for _ in range(256):
        offset = rng.randrange(VIRTUAL_SIZE)
        length = rng.randrange(1, min(6 * STRIPE_SIZE, VIRTUAL_SIZE - offset) + 1)
        assert_logical_read(fh, expected, offset, length)
