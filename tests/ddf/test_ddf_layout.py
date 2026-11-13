from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from dissect.volume.ddf.c_ddf import c_ddf
from dissect.volume.ddf.ddf import DDFVirtualDisk, _convert_raid_layout
from dissect.volume.exceptions import DDFLayoutError
from dissect.volume.raid.raid import PhysicalDisk
from dissect.volume.raid.stream import Layout, Level

BLOCK_SIZE = 512
STRIPE_SIZE = 512
NO_SECONDARY = 0xFF


def make_vdcr(
    prl: int,
    rlq: int,
    srl: int = NO_SECONDARY,
    pec: int | None = None,
    sec: int = 1,
    ses: int = 0,
    strip_size: int = 0,
) -> SimpleNamespace:
    """Build the VirtualDiskConfigurationRecord view that DDFVirtualDisk depends on."""
    return SimpleNamespace(
        primary_raid_level=prl,
        raid_level_qualifier=rlq,
        secondary_raid_level=srl,
        primary_element_count=pec,
        secondary_element_count=sec,
        secondary_element_seq=ses,
        strip_size=strip_size,
        block_size=BLOCK_SIZE,
        size=4,
    )


def make_virtual_disks(disk_patterns: list[bytes], vdcr: SimpleNamespace) -> DDFVirtualDisk:
    """Construct a DDFVirtualDisk backed by small in-memory disks with distinguishable content."""
    disks = {
        idx: (0, PhysicalDisk(io.BytesIO(pattern * BLOCK_SIZE), 0, BLOCK_SIZE))
        for idx, pattern in enumerate(disk_patterns)
    }
    if vdcr.size == 1:
        vdcr.size = len(disk_patterns)
    vdr = SimpleNamespace(name="synthetic", guid=b"\x00" * 24)
    return DDFVirtualDisk(vdcr, vdr, disks)


def pattern_disks(count: int) -> list[bytes]:
    return [bytes([ord("A") + idx]) for idx in range(count)]


@pytest.mark.parametrize(
    ("prl", "rlq", "srl", "pec", "sec", "level", "layout", "num_disks"),
    [
        pytest.param(
            c_ddf.DDF_VDCR_RAID0,
            c_ddf.DDF_VDCR_RAID0_SIMPLE,
            NO_SECONDARY,
            3,
            1,
            Level.RAID0,
            None,
            3,
            id="raid0-simple",
        ),
        pytest.param(
            c_ddf.DDF_VDCR_RAID1,
            c_ddf.DDF_VDCR_RAID1_SIMPLE,
            NO_SECONDARY,
            2,
            1,
            Level.RAID1,
            None,
            2,
            id="raid1-simple",
        ),
        pytest.param(
            c_ddf.DDF_VDCR_RAID1,
            c_ddf.DDF_VDCR_RAID1_MULTI,
            NO_SECONDARY,
            3,
            1,
            Level.RAID1,
            None,
            3,
            id="raid1-multi",
        ),
        pytest.param(
            c_ddf.DDF_VDCR_RAID1,
            c_ddf.DDF_VDCR_RAID1_SIMPLE,
            c_ddf.DDF_VDCR_2SPANNED,
            2,
            2,
            Level.RAID10,
            0x102,
            4,
            id="raid1-spanned-nested",
        ),
        pytest.param(
            c_ddf.DDF_VDCR_RAID1E,
            c_ddf.DDF_VDCR_RAID1E_ADJACENT,
            NO_SECONDARY,
            4,
            1,
            Level.RAID10,
            0x102,
            4,
            id="raid1e-adjacent",
        ),
        pytest.param(
            c_ddf.DDF_VDCR_RAID1E,
            c_ddf.DDF_VDCR_RAID1E_OFFSET,
            NO_SECONDARY,
            4,
            1,
            Level.RAID10,
            0x201,
            4,
            id="raid1e-offset",
        ),
    ],
)
def test_convert_supported_layouts(
    prl: int, rlq: int, srl: int, pec: int, sec: int, level: Level, layout: int | None, num_disks: int
) -> None:
    assert _convert_raid_layout(prl, rlq, srl, pec, sec) == (level, layout, num_disks)


def test_raid0_preserves_member_order() -> None:
    vdcr = make_vdcr(c_ddf.DDF_VDCR_RAID0, c_ddf.DDF_VDCR_RAID0_SIMPLE, pec=3)
    vd = make_virtual_disks(pattern_disks(3), vdcr)

    assert vd.level == Level.RAID0
    assert vd.layout is None
    assert vd.num_disks == 3

    fh = vd.open()
    assert fh.read(STRIPE_SIZE) == b"A" * STRIPE_SIZE
    assert fh.read(STRIPE_SIZE) == b"B" * STRIPE_SIZE
    assert fh.read(STRIPE_SIZE) == b"C" * STRIPE_SIZE


@pytest.mark.parametrize(
    ("rlq", "pec"),
    [
        pytest.param(c_ddf.DDF_VDCR_RAID1_SIMPLE, 2, id="simple"),
        pytest.param(c_ddf.DDF_VDCR_RAID1_MULTI, 3, id="multi"),
    ],
)
def test_raid1_reads_single_mirror_copy(rlq: int, pec: int) -> None:
    vdcr = make_vdcr(c_ddf.DDF_VDCR_RAID1, rlq, pec=pec)
    vd = make_virtual_disks(pattern_disks(pec), vdcr)

    assert vd.level == Level.RAID1
    assert vd.layout is None
    assert vd.num_disks == pec

    fh = vd.open()
    assert fh.read(STRIPE_SIZE) == b"A" * STRIPE_SIZE


def test_nested_raid10_preserves_member_order() -> None:
    vdcr = make_vdcr(
        c_ddf.DDF_VDCR_RAID1,
        c_ddf.DDF_VDCR_RAID1_SIMPLE,
        srl=c_ddf.DDF_VDCR_2SPANNED,
        pec=2,
        sec=2,
    )
    vd = make_virtual_disks(pattern_disks(4), vdcr)

    assert vd.level == Level.RAID10
    assert vd.layout == 0x102
    assert vd.num_disks == 4

    fh = vd.open()
    assert fh.read(STRIPE_SIZE) == b"A" * STRIPE_SIZE
    assert fh.read(STRIPE_SIZE) == b"C" * STRIPE_SIZE


def test_raid1e_adjacent_and_offset_are_not_interchangeable() -> None:
    # These two DDF combinations share PRL=0x11 and the same member set, but the different RLQs map to
    # different RAID10 layouts and therefore reconstruct a different logical member order.
    patterns = pattern_disks(4)

    adjacent = make_virtual_disks(
        patterns, make_vdcr(c_ddf.DDF_VDCR_RAID1E, c_ddf.DDF_VDCR_RAID1E_ADJACENT, pec=4)
    )
    offset = make_virtual_disks(
        patterns, make_vdcr(c_ddf.DDF_VDCR_RAID1E, c_ddf.DDF_VDCR_RAID1E_OFFSET, pec=4)
    )

    assert adjacent.level == offset.level == Level.RAID10
    assert adjacent.layout == 0x102
    assert offset.layout == 0x201

    adjacent_fh = adjacent.open()
    offset_fh = offset.open()
    # First stripe happens to land on the same member for both layouts.
    assert adjacent_fh.read(STRIPE_SIZE) == b"A" * STRIPE_SIZE
    assert offset_fh.read(STRIPE_SIZE) == b"A" * STRIPE_SIZE
    # The second stripe proves the order differs: adjacent jumps to the second mirrored pair (C),
    # while offset walks the members in order (B).
    assert adjacent_fh.read(STRIPE_SIZE) == b"C" * STRIPE_SIZE
    assert offset_fh.read(STRIPE_SIZE) == b"B" * STRIPE_SIZE


def test_nested_striped_and_spanned_are_not_interchangeable_as_contracts() -> None:
    # Both secondary levels currently resolve through the generic RAID10 engine, but they remain distinct
    # DDF combinations: the striped secondary level is explicitly represented while the mirrored one is
    # rejected instead of being silently treated as an equivalent mapping.
    striped = _convert_raid_layout(
        c_ddf.DDF_VDCR_RAID1, c_ddf.DDF_VDCR_RAID1_SIMPLE, c_ddf.DDF_VDCR_2STRIPED, 2, 2
    )
    assert striped == (Level.RAID10, 0x102, 4)

    with pytest.raises(DDFLayoutError) as exc_info:
        _convert_raid_layout(
            c_ddf.DDF_VDCR_RAID1, c_ddf.DDF_VDCR_RAID1_SIMPLE, c_ddf.DDF_VDCR_2MIRRORED, 2, 2
        )
    assert exc_info.value.secondary_raid_level == c_ddf.DDF_VDCR_2MIRRORED


@pytest.mark.parametrize(
    ("kwargs", "message_part"),
    [
        pytest.param(
            dict(prl=c_ddf.DDF_VDCR_RAID0, rlq=0x09, pec=3),
            "Unsupported or contradictory",
            id="unknown-qualifier",
        ),
        pytest.param(
            dict(prl=0x77, rlq=c_ddf.DDF_VDCR_RAID0_SIMPLE, pec=3),
            "Unsupported or contradictory",
            id="unknown-primary-level",
        ),
        pytest.param(
            dict(prl=c_ddf.DDF_VDCR_RAID1, rlq=c_ddf.DDF_VDCR_RAID1_SIMPLE, pec=3),
            "does not match the primary element count",
            id="raid1-simple-wrong-count",
        ),
        pytest.param(
            dict(prl=c_ddf.DDF_VDCR_RAID1, rlq=c_ddf.DDF_VDCR_RAID1_MULTI, srl=NO_SECONDARY, pec=2, sec=2),
            "nested DDF RAID layout",
            id="nested-multi-wrong-count",
        ),
        pytest.param(
            dict(prl=c_ddf.DDF_VDCR_RAID0, rlq=c_ddf.DDF_VDCR_RAID0_SIMPLE, srl=c_ddf.DDF_VDCR_2STRIPED, pec=3),
            "Secondary RAID level is set",
            id="secondary-level-without-elements",
        ),
        pytest.param(
            dict(prl=c_ddf.DDF_VDCR_RAID0, rlq=c_ddf.DDF_VDCR_RAID0_SIMPLE, pec=0),
            "Invalid primary element count",
            id="zero-primary-elements",
        ),
    ],
)
def test_invalid_layouts_raise_with_raw_fields(kwargs: dict[str, Any], message_part: str) -> None:
    defaults = dict(srl=NO_SECONDARY, pec=None, sec=1, ses=0)
    defaults.update(kwargs)
    if defaults["pec"] is None:
        defaults["pec"] = 2

    with pytest.raises(DDFLayoutError) as exc_info:
        _convert_raid_layout(
            defaults["prl"],
            defaults["rlq"],
            defaults["srl"],
            defaults["pec"],
            defaults["sec"],
            defaults["ses"],
        )

    error = exc_info.value
    assert message_part in str(error)
    assert error.primary_raid_level == defaults["prl"]
    assert error.raid_level_qualifier == defaults["rlq"]
    assert error.secondary_raid_level == defaults["srl"]
    assert error.primary_element_count == defaults["pec"]
    assert error.secondary_element_count == defaults["sec"]
    assert error.secondary_element_seq == defaults["ses"]


def test_virtual_disk_construction_rejects_invalid_layout() -> None:
    vdcr = make_vdcr(c_ddf.DDF_VDCR_RAID1, c_ddf.DDF_VDCR_RAID1_MULTI, pec=2)
    with pytest.raises(DDFLayoutError) as exc_info:
        make_virtual_disks(pattern_disks(2), vdcr)

    assert exc_info.value.primary_raid_level == c_ddf.DDF_VDCR_RAID1
    assert exc_info.value.raid_level_qualifier == c_ddf.DDF_VDCR_RAID1_MULTI
    assert exc_info.value.primary_element_count == 2
