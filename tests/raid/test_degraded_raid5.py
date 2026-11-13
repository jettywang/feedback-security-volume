from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import pytest

from dissect.volume.exceptions import RAIDError, RAIDStripeReadError
from dissect.volume.raid.raid import PhysicalDisk, VirtualDisk
from dissect.volume.raid.stream import Layout, Level, RAID456Stream

RAID_DISKS = 4
CHUNK_SIZE = 64
DATA_DISKS = RAID_DISKS - 1
STRIPES = RAID_DISKS * 2
VOLUME_SIZE = STRIPES * DATA_DISKS * CHUNK_SIZE


class ShortReadIO(BytesIO):
    """A file-like object that silently returns EOF, emulating a member that cannot serve its data."""

    def read(self, size: int = -1) -> bytes:
        return b""


def expected_byte(logical_offset: int) -> int:
    """Deterministic, non-zero logical content: distinct and predictable per byte."""
    return ((logical_offset * 7 + 11) & 0xFF) or 0x5A


def build_volume() -> bytes:
    return bytes(expected_byte(offset) for offset in range(VOLUME_SIZE))


def parity_position(layout: Layout, stripe: int) -> int:
    if layout in (Layout.LEFT_ASYMMETRIC, Layout.LEFT_SYMMETRIC):
        return DATA_DISKS - (stripe % RAID_DISKS)
    return stripe % RAID_DISKS


def data_position(layout: Layout, stripe: int, dd_idx: int) -> int:
    pd_idx = parity_position(layout, stripe)
    if layout in (Layout.LEFT_SYMMETRIC, Layout.RIGHT_SYMMETRIC):
        return (pd_idx + 1 + dd_idx) % RAID_DISKS

    # Asymmetric ("restart") layouts: data disks keep their natural order around the parity slot.
    return dd_idx if dd_idx < pd_idx else dd_idx + 1


def build_members(layout: Layout) -> list[bytearray]:
    """Construct the RAID5 member images with correct, independently derived P parity."""
    volume = build_volume()
    members = [bytearray(STRIPES * CHUNK_SIZE) for _ in range(RAID_DISKS)]

    for stripe in range(STRIPES):
        pd_idx = parity_position(layout, stripe)
        parity = bytearray(CHUNK_SIZE)
        for dd_idx in range(DATA_DISKS):
            disk_idx = data_position(layout, stripe, dd_idx)
            logical_offset = (stripe * DATA_DISKS + dd_idx) * CHUNK_SIZE
            chunk = volume[logical_offset : logical_offset + CHUNK_SIZE]
            members[disk_idx][stripe * CHUNK_SIZE : (stripe + 1) * CHUNK_SIZE] = chunk
            for pos, value in enumerate(chunk):
                parity[pos] ^= value
        members[pd_idx][stripe * CHUNK_SIZE : (stripe + 1) * CHUNK_SIZE] = parity

    return members


def make_virtual_disk(
    layout: Layout, member_data: list[bytearray | BytesIO], num_disks: int = RAID_DISKS
) -> VirtualDisk:
    disk_map = {}
    for disk_idx, data in enumerate(member_data):
        if data is None:
            continue
        raw = data if isinstance(data, BytesIO) else BytesIO(bytes(data))
        disk_map[disk_idx] = (0, PhysicalDisk(raw, 0, len(raw.getvalue())))

    return VirtualDisk(
        name="degraded-test",
        uuid="00000000-0000-0000-0000-000000000000",
        size=VOLUME_SIZE,
        level=Level.RAID5,
        layout=layout,
        stripe_size=CHUNK_SIZE,
        num_disks=num_disks,
        disk_map=disk_map,
    )


def open_degraded(layout: Layout, missing: set[int]) -> tuple[RAID456Stream, bytes]:
    members = build_members(layout)
    surviving = [data if disk_idx not in missing else None for disk_idx, data in enumerate(members)]
    stream = RAID456Stream(make_virtual_disk(layout, surviving))
    return stream, build_volume()


ALL_LAYOUTS = [
    Layout.LEFT_ASYMMETRIC,
    Layout.RIGHT_ASYMMETRIC,
    Layout.LEFT_SYMMETRIC,
    Layout.RIGHT_SYMMETRIC,
]


@pytest.mark.parametrize("layout", ALL_LAYOUTS)
def test_healthy_read_matches_volume(layout: Layout) -> None:
    stream, volume = open_degraded(layout, set())
    assert stream.read() == volume
    assert stream.reconstructed_ranges == []


@pytest.mark.parametrize("layout", ALL_LAYOUTS)
@pytest.mark.parametrize("missing_disk", range(RAID_DISKS))
def test_single_missing_disk_rebuilds_byte_for_byte(layout: Layout, missing_disk: int) -> None:
    stream, volume = open_degraded(layout, {missing_disk})

    # Whole-volume, cross-stripe read must be reconstructed exactly, never zero padded.
    recovered = stream.read()
    assert recovered == volume
    assert b"\x00" not in recovered or b"\x00" in volume

    # Every logical stripe whose data lived on the missing disk must be marked reconstructed.
    bitmap = stream.reconstructed_bitmap(0, VOLUME_SIZE)
    assert len(bitmap) == VOLUME_SIZE
    for stripe in range(STRIPES):
        for dd_idx in range(DATA_DISKS):
            logical_offset = (stripe * DATA_DISKS + dd_idx) * CHUNK_SIZE
            disk_idx = data_position(layout, stripe, dd_idx)
            for offset in range(logical_offset, logical_offset + CHUNK_SIZE):
                assert bool(bitmap[offset]) is (disk_idx == missing_disk)


@pytest.mark.parametrize("layout", [Layout.LEFT_SYMMETRIC, Layout.RIGHT_SYMMETRIC])
@pytest.mark.parametrize("missing_disk", range(RAID_DISKS))
def test_missing_data_and_missing_parity_are_both_recoverable(layout: Layout, missing_disk: int) -> None:
    stream, volume = open_degraded(layout, {missing_disk})

    # A stripe where the missing member holds a data chunk: that chunk must be rebuilt from P.
    data_stripe, data_dd = next(
        (stripe, dd_idx)
        for stripe in range(STRIPES)
        for dd_idx in range(DATA_DISKS)
        if data_position(layout, stripe, dd_idx) == missing_disk
    )
    # A stripe where the missing member holds parity: data is read directly from surviving members.
    parity_stripe = next(stripe for stripe in range(STRIPES) if parity_position(layout, stripe) == missing_disk)
    parity_dd = 0
    assert data_position(layout, parity_stripe, parity_dd) != missing_disk

    data_offset = (data_stripe * DATA_DISKS + data_dd) * CHUNK_SIZE
    parity_data_offset = (parity_stripe * DATA_DISKS + parity_dd) * CHUNK_SIZE

    assert stream.readoffset(data_offset, CHUNK_SIZE) == volume[data_offset : data_offset + CHUNK_SIZE]
    assert stream.is_reconstructed(data_offset, CHUNK_SIZE)

    assert (
        stream.readoffset(parity_data_offset, CHUNK_SIZE)
        == volume[parity_data_offset : parity_data_offset + CHUNK_SIZE]
    )
    assert not stream.is_reconstructed(parity_data_offset, CHUNK_SIZE)


@pytest.mark.parametrize("layout", [Layout.LEFT_SYMMETRIC, Layout.RIGHT_SYMMETRIC])
def test_unaligned_cross_stripe_read(layout: Layout) -> None:
    missing_disk = 1
    stream, volume = open_degraded(layout, {missing_disk})

    start = CHUNK_SIZE // 2
    length = CHUNK_SIZE * DATA_DISKS * 3 + 17
    recovered = stream.readoffset(start, length)
    assert recovered == volume[start : start + length]

    bitmap = stream.reconstructed_bitmap(start, length)
    for relative, logical in enumerate(range(start, start + length)):
        stripe, dd_idx = divmod(logical // CHUNK_SIZE, DATA_DISKS)
        assert bool(bitmap[relative]) is (data_position(layout, stripe, dd_idx) == missing_disk)


@pytest.mark.parametrize("layout", ALL_LAYOUTS)
def test_two_missing_disks_raises_with_evidence(layout: Layout) -> None:
    members = build_members(layout)
    surviving = [None if disk_idx in (0, 2) else data for disk_idx, data in enumerate(members)]

    with pytest.raises(RAIDError):
        RAID456Stream(make_virtual_disk(layout, surviving))


@pytest.mark.parametrize("layout", [Layout.LEFT_SYMMETRIC, Layout.RIGHT_SYMMETRIC])
def test_second_failure_during_rebuild_raises_with_evidence(layout: Layout) -> None:
    # One disk is absent; a second member is present but cannot serve its chunks.
    missing_disk = 0
    failed_disk = 2
    members = build_members(layout)
    surviving: list[BinaryIO | None] = []
    for disk_idx, data in enumerate(members):
        if disk_idx == missing_disk:
            surviving.append(None)
        elif disk_idx == failed_disk:
            surviving.append(ShortReadIO(bytes(data)))
        else:
            surviving.append(BytesIO(bytes(data)))

    stream = RAID456Stream(make_virtual_disk(layout, surviving))

    # Locate a stripe where the absent member holds a data chunk: reading it forces a rebuild that
    # also touches the short-reading member, producing evidence for both failures.
    target_stripe, target_dd = next(
        (stripe, dd_idx)
        for stripe in range(STRIPES)
        for dd_idx in range(DATA_DISKS)
        if data_position(layout, stripe, dd_idx) == missing_disk and parity_position(layout, stripe) != missing_disk
    )
    target_offset = (target_stripe * DATA_DISKS + target_dd) * CHUNK_SIZE

    with pytest.raises(RAIDStripeReadError) as exc_info:
        stream.readoffset(target_offset, CHUNK_SIZE)

    error = exc_info.value
    assert error.stripe_number == target_stripe
    disk_indices = {item.disk_idx for item in error.evidence}
    assert missing_disk in disk_indices
    assert failed_disk in disk_indices
    for item in error.evidence:
        assert item.expected_length == CHUNK_SIZE or item.actual_length is None
    short_item = next(item for item in error.evidence if item.disk_idx == failed_disk)
    assert short_item.actual_length == 0
    assert "short read" in short_item.reason


@pytest.mark.parametrize("layout", [Layout.LEFT_SYMMETRIC, Layout.RIGHT_SYMMETRIC])
def test_short_read_is_not_zero_padded(layout: Layout) -> None:
    # All members are present, but one truncates reads: this must surface as an error, not zeros.
    failed_disk = 1
    members = build_members(layout)
    surviving = [
        ShortReadIO(bytes(data)) if disk_idx == failed_disk else BytesIO(bytes(data))
        for disk_idx, data in enumerate(members)
    ]
    stream = RAID456Stream(make_virtual_disk(layout, surviving))

    target_stripe, target_dd = next(
        (stripe, dd_idx)
        for stripe in range(STRIPES)
        for dd_idx in range(DATA_DISKS)
        if data_position(layout, stripe, dd_idx) == failed_disk
    )
    target_offset = (target_stripe * DATA_DISKS + target_dd) * CHUNK_SIZE

    with pytest.raises(RAIDStripeReadError) as exc_info:
        stream.readoffset(target_offset, CHUNK_SIZE)

    item = exc_info.value.evidence[0]
    assert item.disk_idx == failed_disk
    assert item.expected_length == CHUNK_SIZE
    assert item.actual_length == 0
    assert item.member_offset == target_stripe * CHUNK_SIZE
