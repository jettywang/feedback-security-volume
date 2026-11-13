from __future__ import annotations

import io
from datetime import datetime, timezone

from dissect.volume.vinum.c_vinum import c_vinum
from dissect.volume.vinum.config import (
    VolumeAvailability,
    analyze_vinum_configs,
    parse_vinum_config_with_issues,
)
from dissect.volume.vinum.vinum import Vinum

CONF_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_disk(name: bytes, config: bytes, *, epoch: int = 1, active: bool = True) -> bytes:
    """Build a minimal Vinum physical disk image with the given config."""
    header = c_vinum.header()
    header.magic = c_vinum.GV_MAGIC if active else c_vinum.GV_NOMAGIC
    header.config_length = len(config)
    header.label.name = name
    header.label.last_update.sec = epoch
    header.label.drive_size = 0

    data = bytearray(c_vinum.GV_DATA_START)
    data[c_vinum.GV_HDR_OFFSET : c_vinum.GV_HDR_OFFSET + len(bytes(header))] = bytes(header)
    data[c_vinum.GV_CFG_OFFSET : c_vinum.GV_CFG_OFFSET + len(config)] = config
    return bytes(data)


def analyze(config: bytes, drives: list[bytes] | None = None, disk: bytes = b"disk0"):
    parsed = parse_vinum_config_with_issues(CONF_TS, config, disk=disk)
    return analyze_vinum_configs(parsed, drives=drives)


def codes(issues) -> list[str]:
    return [issue.code for issue in issues]


def test_parse_keeps_source_line() -> None:
    config = (
        b"volume vol state up\n"
        b"plex name vol.p0 org concat vol vol state up\n"
        b"sd name vol.p0.s0 drive disk0 len 4s driveoffset 265s plex vol.p0 plexoffset 0s state up\n"
    )
    parsed = parse_vinum_config_with_issues(CONF_TS, config, disk=b"disk0")

    assert parsed.volumes[0].source.line == 1
    assert parsed.volumes[0].source.raw == b"volume vol state up"
    assert parsed.plexes[0].source.line == 2
    assert parsed.sds[0].source.line == 3
    assert parsed.sds[0].source.disk == b"disk0"
    assert parsed.sds[0].source.offset == len(config.splitlines(keepends=True)[0]) + len(
        config.splitlines(keepends=True)[1]
    )


def test_parse_failure_keeps_line_evidence() -> None:
    config = b"volume vol state up\nplex name vol.p0 org concat vol vol state up\nsd name vol.p0.s0 drive\n"
    parsed = parse_vinum_config_with_issues(CONF_TS, config)

    assert len(parsed.volumes) == 1
    assert len(parsed.plexes) == 1
    assert parsed.sds == []
    assert "invalid_config_line" in codes(parsed.issues)
    assert parsed.issues[0].line == 3


def test_implicit_links_and_healthy_volume_isolation() -> None:
    config = (
        b"volume healthy state up\n"
        b"volume broken state up\n"
        b"plex name healthy.p0 org concat state up\n"
        b"plex name broken.p0 org concat state up\n"
        b"sd name healthy.p0.s0 drive disk0 len 4s driveoffset 265s plexoffset 0s state up\n"
        b"sd name healthy.p0.s1 drive disk1 len 4s driveoffset 265s plexoffset 4s state up\n"
        b"sd name broken.p0.s0 drive disk0 len 4s driveoffset 265s plexoffset 0s state up\n"
        b"sd name broken.p0.s1 drive disk2 len 4s driveoffset 265s plexoffset 8s state up\n"
    )
    diagnostics = analyze(config, drives=[b"disk0", b"disk1"])

    assert diagnostics.readable_volumes == [b"healthy"]
    assert diagnostics.evidence_only_volumes == [b"broken"]

    healthy = diagnostics.assessment(b"healthy")
    assert healthy is not None and healthy.status is VolumeAvailability.READABLE
    assert [plex.name for plex in healthy.plexes] == [b"healthy.p0"]
    assert "implicit_plex_volume" in codes(healthy.issues)
    assert "implicit_sd_plex" in codes(healthy.issues)

    broken = diagnostics.assessment(b"broken")
    assert broken is not None and broken.status is VolumeAvailability.EVIDENCE_ONLY
    missing = [issue for issue in broken.issues if issue.code == "sd_missing_drive"]
    assert len(missing) == 1
    assert missing[0].drive == b"disk2"
    assert missing[0].line == 8


def test_mirror_plex_and_duplicate_names() -> None:
    config = (
        b"volume mirror state up\n"
        b"volume mirror state up\n"
        b"plex name mirror.p0 org concat vol mirror state up\n"
        b"plex name mirror.p1 org concat vol mirror state up\n"
        b"sd name mirror.p0.s0 drive disk0 len 4s driveoffset 265s plex mirror.p0 plexoffset 0s state up\n"
        b"sd name mirror.p1.s0 drive disk1 len 8s driveoffset 265s plex mirror.p1 plexoffset 0s state up\n"
    )
    diagnostics = analyze(config, drives=[b"disk0", b"disk1"])

    assessment = diagnostics.assessment(b"mirror")
    assert assessment is not None
    assert assessment.status is VolumeAvailability.EVIDENCE_ONLY
    assert any(issue.code == "mirror_plex_size_mismatch" for issue in assessment.issues)
    duplicate = [issue for issue in diagnostics.issues if issue.code == "duplicate_volume"]
    assert len(duplicate) == 1
    assert duplicate[0].line == 1
    assert len(duplicate[0].sources) == 2


def test_overlapping_ranges() -> None:
    config = (
        b"volume good state up\n"
        b"volume overlap state up\n"
        b"plex name good.p0 org concat vol good state up\n"
        b"plex name overlap.p0 org concat vol overlap state up\n"
        b"sd name good.p0.s0 drive disk0 len 4s driveoffset 500s plex good.p0 plexoffset 0s state up\n"
        b"sd name overlap.p0.s0 drive disk0 len 4s driveoffset 265s plex overlap.p0 plexoffset 0s state up\n"
        b"sd name overlap.p0.s1 drive disk0 len 4s driveoffset 267s plex overlap.p0 plexoffset 4s state up\n"
    )
    diagnostics = analyze(config, drives=[b"disk0"])

    assert diagnostics.readable_volumes == [b"good"]
    overlap = diagnostics.assessment(b"overlap")
    assert overlap is not None and overlap.status is VolumeAvailability.EVIDENCE_ONLY
    drive_overlap = [issue for issue in overlap.issues if issue.code == "sd_driveoffset_overlap"]
    assert len(drive_overlap) == 2
    assert {issue.line for issue in drive_overlap} == {6, 7}
    assert {issue.drive for issue in drive_overlap} == {b"disk0"}


def test_plex_ownership_mismatch() -> None:
    config = (
        b"volume a state up\n"
        b"volume b state up\n"
        b"plex name a.p0 org concat vol b state up\n"
        b"sd name a.p0.s0 drive disk0 len 4s driveoffset 265s plex a.p0 plexoffset 0s state up\n"
    )
    diagnostics = analyze(config, drives=[b"disk0"])

    assert diagnostics.readable_volumes == []
    assert diagnostics.evidence_only_volumes == [b"a", b"b"]
    mismatch = [issue for issue in diagnostics.issues if issue.code == "plex_volume_ownership_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0].line == 3
    assert mismatch[0].volume == b"b"
    assert mismatch[0].plex == b"a.p0"


def test_down_volume_without_plex() -> None:
    config = b"volume down state down\n"
    diagnostics = analyze(config)

    assessment = diagnostics.assessment(b"down")
    assert assessment is not None
    assert assessment.status is VolumeAvailability.EVIDENCE_ONLY
    assert assessment.reason == "volume is down"


def test_unreadable_sd_state() -> None:
    config = (
        b"volume vol state up\n"
        b"plex name vol.p0 org concat vol vol state up\n"
        b"sd name vol.p0.s0 drive disk0 len 4s driveoffset 265s plex vol.p0 plexoffset 0s state down\n"
    )
    diagnostics = analyze(config, drives=[b"disk0"])

    assert diagnostics.readable_volumes == []
    unreadable = [issue for issue in diagnostics.issues if issue.code == "sd_unreadable_state"]
    assert len(unreadable) == 1
    assert unreadable[0].line == 3
    assert unreadable[0].sd == b"vol.p0.s0"


def test_original_config_is_not_modified() -> None:
    config = (
        b"volume vol state up\n"
        b"plex name vol.p0 org concat vol vol state up\n"
        b"sd name vol.p0.s0 drive disk0 len 4s driveoffset 265s plex vol.p0 plexoffset 0s state up\n"
    )
    original = bytes(config)
    diagnostics = analyze(config, drives=[b"disk0"])

    assert diagnostics.readable_volumes == [b"vol"]
    assert config == original

    image = make_disk(b"disk0", config)
    before = image[:]
    Vinum(io.BytesIO(image))

    assert image == before


def test_virtual_disks_only_for_readable_volumes() -> None:
    config = (
        b"volume good state up\n"
        b"volume bad state up\n"
        b"plex name good.p0 org concat vol good state up\n"
        b"plex name bad.p0 org concat vol bad state up\n"
        b"sd name good.p0.s0 drive disk0 len 4s driveoffset 265s plex good.p0 plexoffset 0s state up\n"
        b"sd name good.p0.s1 drive disk1 len 4s driveoffset 265s plex good.p0 plexoffset 4s state up\n"
        b"sd name bad.p0.s0 drive disk0 len 4s driveoffset 265s plex bad.p0 plexoffset 0s state up\n"
        b"sd name bad.p0.s1 drive disk2 len 4s driveoffset 265s plex bad.p0 plexoffset 4s state up\n"
    )
    disks = [io.BytesIO(make_disk(b"disk0", config)), io.BytesIO(make_disk(b"disk1", config))]

    vinum = Vinum(disks)

    assert [vd.name for vd in vinum.configurations[0].virtual_disks] == ["good"]
    assert vinum.diagnostics.evidence_only_volumes == [b"bad"]

    assert vinum.configurations[0].virtual_disks[0].size == 4096


def test_existing_virtual_disks_are_unchanged(vinum_mirror) -> None:
    vinum = Vinum(vinum_mirror)

    assert len(vinum.configurations[0].virtual_disks) == 1
    assert vinum.configurations[0].virtual_disks[0].name == "my-mirror-vol"
    assert set(vinum.diagnostics.readable_volumes + vinum.diagnostics.evidence_only_volumes) == {
        b"my-mirror-vol",
        b"my-striped-vol",
        b"my-raid5-vol",
        b"my-concat-vol",
        b"my-stripedmirror-vol",
    }


def test_mirror_survives_one_down_plex() -> None:
    config = (
        b"volume mirror state up\n"
        b"plex name mirror.p0 org concat vol mirror state up\n"
        b"plex name mirror.p1 org concat vol mirror state down\n"
        b"sd name mirror.p0.s0 drive disk0 len 4s driveoffset 265s plex mirror.p0 plexoffset 0s state up\n"
        b"sd name mirror.p1.s0 drive disk1 len 4s driveoffset 265s plex mirror.p1 plexoffset 0s state up\n"
    )
    diagnostics = analyze(config, drives=[b"disk0", b"disk1"])

    assessment = diagnostics.assessment(b"mirror")
    assert assessment is not None
    assert assessment.status is VolumeAvailability.READABLE
    assert [plex.name for plex in assessment.plexes] == [b"mirror.p0"]
    assert "plex_down" in codes(assessment.issues)
    assert "plex_unusable" in codes(assessment.issues)


def test_ordinary_overlap_keeps_independent_volume() -> None:
    config = (
        b"volume healthy state up\n"
        b"volume overlap state up\n"
        b"plex name healthy.p0 org concat vol healthy state up\n"
        b"plex name overlap.p0 org concat vol overlap state up\n"
        b"sd name healthy.p0.s0 drive disk0 len 4s driveoffset 500s plex healthy.p0 plexoffset 0s state up\n"
        b"sd name overlap.p0.s0 drive disk0 len 4s driveoffset 265s plex overlap.p0 plexoffset 0s state up\n"
        b"sd name overlap.p0.s1 drive disk0 len 4s driveoffset 267s plex overlap.p0 plexoffset 4s state up\n"
    )
    diagnostics = analyze(config, drives=[b"disk0"])

    # The overlap is inside overlap.p0, so the unrelated, physically disjoint
    # healthy volume remains readable.
    assert b"healthy" in diagnostics.readable_volumes
    assert b"overlap" in diagnostics.evidence_only_volumes


def test_conflicting_duplicates_can_not_be_promoted() -> None:
    config = (
        b"volume dup state up\n"
        b"volume dup state down\n"
        b"plex name dup.p0 org concat vol dup state up\n"
        b"sd name dup.p0.s0 drive disk0 len 4s driveoffset 265s plex dup.p0 plexoffset 0s state up\n"
        b"volume healthy state up\n"
        b"plex name healthy.p0 org concat vol healthy state up\n"
        b"sd name healthy.p0.s0 drive disk1 len 4s driveoffset 265s plex healthy.p0 plexoffset 0s state up\n"
    )
    diagnostics = analyze(config, drives=[b"disk0", b"disk1"])

    assert diagnostics.readable_volumes == [b"healthy"]
    assert diagnostics.evidence_only_volumes == [b"dup"]
    assert "duplicate_volume_conflict" in codes(diagnostics.assessment(b"dup").issues)
