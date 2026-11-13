from __future__ import annotations

import itertools

import pytest

from dissect.volume.exceptions import LVM2Error
from dissect.volume.lvm import LVM2
from dissect.volume.lvm.physical import LVM2Device
from tests.lvm._image_builder import RawLocationOverride, build_metadata, build_pv

VG_NAME = "vg_test"
VG_ID = "VGID0000000000000000000000000001"
PV0 = "pv000000000000000000000000000aa"
PV1 = "pv100000000000000000000000000bb"
PV2 = "pv200000000000000000000000000cc"


def _config(seqno: int, pvs: list[tuple[str, str]], lvs: dict[str, str] | None = None) -> str:
    return build_metadata(VG_NAME, VG_ID, seqno, pvs, lvs)


def _open(images: list, reverse: bool = False) -> LVM2:
    handles = [image.open() for image in images]
    return LVM2(handles[::-1] if reverse else handles)


def test_minimal_mirror_order_independent() -> None:
    """Two PVs holding the same config must yield the same result regardless of input ordering."""
    pvs = [("pv0", PV0), ("pv1", PV1)]
    text = _config(1, pvs, {"lv0": "pv0", "lv1": "pv1"})
    images = [build_pv(PV0, [text]), build_pv(PV1, [text])]

    forward = _open(images)
    reverse = _open(images, reverse=True)
    permutations = [LVM2([img.open() for img in perm]) for perm in itertools.permutations(images)]

    for lvm in [forward, reverse, *permutations]:
        assert lvm.vg.id == VG_ID
        assert lvm.vg.seqno == 1
        assert sorted(lv.name for lv in lvm.vg.lv) == ["lv0", "lv1"]
        assert sorted(lvm.devices) == [PV0, PV1]

    assert forward.metadata_report() == reverse.metadata_report()


def test_highest_seqno_wins_and_older_excluded() -> None:
    """The newest internally consistent generation is selected; the older copy is explicitly excluded."""
    pvs = [("pv0", PV0), ("pv1", PV1)]
    old = _config(1, pvs, {"lv_old": "pv0"})
    new = _config(2, pvs, {"lv_new": "pv1"})

    # pv0 still holds generation 1, pv1 already holds generation 2.
    lvm = _open([build_pv(PV0, [old]), build_pv(PV1, [new])])

    assert lvm.vg.seqno == 2
    assert list(lvm.vg.logical_volumes) == ["lv_new"]

    selected = lvm.selection.selected
    assert selected.selected is True
    assert selected.device_id == PV1

    excluded = lvm.selection.excluded
    assert len(excluded) == 1
    assert excluded[0].device_id == PV0
    assert "older generation" in excluded[0].reason
    # No ghost logical volume from the older generation may appear.
    assert "lv_old" not in lvm.vg.logical_volumes


def test_corrupt_newer_copy_falls_back_to_older() -> None:
    """A checksum-damaged newest copy is rejected and the previous valid generation is used."""
    pvs = [("pv0", PV0), ("pv1", PV1)]
    old = _config(1, pvs, {"lv_old": "pv0"})
    new = _config(2, pvs, {"lv_new": "pv1"})

    lvm = _open(
        [
            build_pv(PV0, [old]),
            build_pv(PV1, [RawLocationOverride(text=new, corrupt_checksum=True)]),
        ]
    )

    assert lvm.vg.seqno == 1
    assert list(lvm.vg.logical_volumes) == ["lv_old"]

    reasons = {(r.device_id, r.reason) for r in lvm.selection.excluded}
    assert any(device == PV1 and "checksum mismatch" in reason for device, reason in reasons)


def test_all_copies_corrupt_raises() -> None:
    """If every copy fails validation a deterministic error is raised instead of guessing."""
    pvs = [("pv0", PV0)]
    text = _config(1, pvs)
    image = build_pv(PV0, [RawLocationOverride(text=text, corrupt_text=True)])

    with pytest.raises(LVM2Error, match="No valid LVM2 metadata"):
        _open([image])


def test_missing_physical_volume_is_detached() -> None:
    """A PV referenced by the config but not supplied remains detached without breaking the others."""
    pvs = [("pv0", PV0), ("pv1", PV1)]
    text = _config(1, pvs, {"lv_on_pv0": "pv0", "lv_on_pv1": "pv1"})
    # Only pv0 is supplied; pv1 is missing from the evidence set.
    lvm = _open([build_pv(PV0, [text])])

    assert lvm.missing_physical_volumes == ["pv1"]
    pv_by_id = {pv.id.replace("-", ""): pv for pv in lvm.vg.pv}
    assert pv_by_id[PV0].dev is not None
    assert pv_by_id[PV1].dev is None

    # The LV backed by the present PV still opens; the missing one raises a clear, deterministic error.
    lvm.vg.logical_volumes["lv_on_pv0"].open().close()
    with pytest.raises(LVM2Error, match="Physical volume not found: pv1"):
        lvm.vg.logical_volumes["lv_on_pv1"].open()


def test_same_seqno_different_content_deterministic() -> None:
    """Conflicting copies with an identical seqno resolve deterministically, never by merging."""
    pvs = [("pv0", PV0), ("pv1", PV1)]
    config_a = _config(3, pvs, {"lv_alpha": "pv0"})
    config_b = _config(3, pvs, {"lv_beta": "pv1"})

    images = [build_pv(PV0, [config_a]), build_pv(PV1, [config_b])]
    first = _open(images)
    second = _open(images, reverse=True)

    # Both orderings pick the exact same single, whole configuration.
    assert first.vg.seqno == second.vg.seqno == 3
    assert list(first.vg.logical_volumes) == list(second.vg.logical_volumes)
    assert len(first.vg.logical_volumes) == 1
    # Exactly one config is used; no combination of lv_alpha + lv_beta exists.
    chosen = next(iter(first.vg.logical_volumes))
    assert chosen in {"lv_alpha", "lv_beta"}

    excluded = first.selection.excluded
    assert any("different content" in record.reason for record in excluded)


def test_same_seqno_quorum_prefers_corroborated_config() -> None:
    """When several PVs agree, the config with the most witnesses wins regardless of argument order."""
    pvs = [("pv0", PV0), ("pv1", PV1), ("pv2", PV2)]
    majority = _config(2, pvs, {"lv_majority": "pv0"})
    minority = _config(2, pvs, {"lv_minority": "pv2"})

    images = [
        build_pv(PV0, [majority]),
        build_pv(PV1, [majority]),
        build_pv(PV2, [minority]),
    ]
    for perm in itertools.permutations(images):
        lvm = LVM2([img.open() for img in perm])
        assert list(lvm.vg.logical_volumes) == ["lv_majority"]
        assert "lv_minority" not in lvm.vg.logical_volumes


def test_report_lists_used_and_excluded_with_reasons() -> None:
    """The report documents the adopted copy and every excluded copy with the decision reason."""
    pvs = [("pv0", PV0), ("pv1", PV1)]
    old = _config(1, pvs, {"lv_old": "pv0"})
    new = _config(2, pvs, {"lv_new": "pv1"})
    lvm = _open([build_pv(PV0, [old]), build_pv(PV1, [new])])

    report = lvm.metadata_report()
    assert "[USED]" in report
    assert "[EXCLUDED]" in report
    assert PV1 in report and PV0 in report
    assert "older generation" in report

    # The structured records mirror the textual report.
    assert lvm.selection.selected.selected is True
    assert all(not record.selected for record in lvm.selection.excluded)
    assert all(record.reason for record in lvm.selection.records)


def test_no_frankenstein_across_generations() -> None:
    """LVs from different generations can never be combined into a non-existent volume set."""
    pvs = [("pv0", PV0), ("pv1", PV1)]
    gen1 = _config(1, pvs, {"only_in_gen1": "pv0"})
    gen2 = _config(2, pvs, {"only_in_gen2": "pv1"})

    lvm = _open([build_pv(PV0, [gen1]), build_pv(PV1, [gen2])])

    assert list(lvm.vg.logical_volumes) == ["only_in_gen2"]
    assert "only_in_gen1" not in lvm.vg.logical_volumes
    # The selected VG object is sourced wholly from one raw metadata dictionary.
    selected_data = lvm.selection.metadata
    vg_section = next(value for key, value in selected_data.items() if isinstance(value, dict))
    assert set(vg_section["logical_volumes"]) == {"only_in_gen2"}


def test_ignored_raw_location_is_skipped() -> None:
    """Raw locations flagged as ignored are never considered, even when listed first."""
    pvs = [("pv0", PV0)]
    valid = _config(1, pvs, {"lv_valid": "pv0"})
    image = build_pv(PV0, [RawLocationOverride(text=valid, ignored=True), valid])

    lvm = _open([image])
    assert list(lvm.vg.logical_volumes) == ["lv_valid"]
    assert any("ignored" in record.reason for record in lvm.selection.excluded)


def test_distinct_volume_groups_are_rejected() -> None:
    """Copies belonging to different VG identities cannot be merged into one Volume Group."""
    other_id = "VGID0000000000000000000000000099"
    pvs = [("pv0", PV0), ("pv1", PV1)]
    config_one = build_metadata(VG_NAME, VG_ID, 1, pvs)
    config_two = build_metadata("vg_other", other_id, 1, pvs)

    images = [build_pv(PV0, [config_one]), build_pv(PV1, [config_two])]
    with pytest.raises(LVM2Error, match="multiple distinct volume groups"):
        _open(images)


def test_multiple_raw_locations_in_one_area() -> None:
    """Several raw locations in a single metadata area are all enumerated and reconciled by seqno."""
    pvs = [("pv0", PV0)]
    gen1 = _config(1, pvs, {"lv_first": "pv0"})
    gen2 = _config(2, pvs, {"lv_second": "pv0"})
    gen3 = _config(3, pvs, {"lv_third": "pv0"})

    device = LVM2Device(build_pv(PV0, [gen1, gen2, gen3]).open())
    assert len(device.metadata_copies) == 3
    assert all(copy.valid for copy in device.metadata_copies)

    lvm = LVM2(device)
    assert lvm.vg.seqno == 3
    assert list(lvm.vg.logical_volumes) == ["lv_third"]
    excluded_seqnos = [record.reason for record in lvm.selection.excluded]
    assert sum("older generation" in reason for reason in excluded_seqnos) == 2
