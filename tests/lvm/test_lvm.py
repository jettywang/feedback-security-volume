from __future__ import annotations

import io
import re
from typing import BinaryIO

import pytest

from dissect.volume.exceptions import LVM2Error
from dissect.volume.lvm import LVM2
from dissect.volume.lvm.c_lvm2 import SECTOR_SIZE, c_lvm
from dissect.volume.lvm.metadata import LVM2Mapping, StripedSegment


def test_lvm(lvm: BinaryIO) -> None:
    vs = LVM2(lvm)

    vg = vs.vg
    assert vg.name == "vg_test"
    assert vg.id == "8HfEjs-9DNH-0dy1-U5u8-EYBF-Vce4-8BcSWU"
    assert len(vg.lv) == 1
    assert len(vg.pv) == 1

    lv = vg.lv[0]
    assert lv.name == "lv_test"
    assert lv.id == "TnYdWo-zRE9-wf2T-5nt0-M1aD-vtoP-fASCxK"
    assert lv.type == "striped"
    assert len(lv.segments) == 1

    seg = lv.segments[0]
    assert isinstance(seg, StripedSegment)
    assert len(seg.stripes) == seg.stripe_count
    assert seg.extent_count == 1
    assert seg.extent_count * vg.extent_size == 8192

    fh = lv.open()
    assert fh.read(1024) == b"\xde\xad\xbe\xef" * 256
    assert fh.read(1024) == b"\x00" * 1024

    vg.pv[0]._dev = None
    with pytest.raises(
        LVM2Error,
        match=re.escape(
            "Physical volume not found: pv0 (id=2Svcy0-cRH2-3Xrz-87Fv-zNUI-9CoI-Ycoyql, device=/dev/loop1)"
        ),
    ):
        fh = lv.open()


def test_lvm_invalid() -> None:
    buf = io.BytesIO(4096 * b"\x00")

    with pytest.raises(LVM2Error):
        LVM2(buf)


def test_lvm_thin(lvm_thin: BinaryIO) -> None:
    lvm = LVM2(lvm_thin)

    for lv_name in ["lv-1", "lv-2"]:
        lv = lvm.vg.logical_volumes[lv_name]
        assert lv.type == "thin"

        fh = lv.open()
        for i in range(1, 513):
            assert fh.read(4096) == i.to_bytes(2, "little") * 2048

    pool = lvm.vg.logical_volumes["data"]
    assert pool.type == "thin-pool"

    with pytest.raises(
        RuntimeError, match="Opening a thin-pool for reading is not possible, use open_pool\\(\\) instead"
    ):
        pool.open()


def test_lvm_mirror(lvm_mirror: list[BinaryIO]) -> None:
    lvm = LVM2(lvm_mirror)
    lv = lvm.vg.logical_volumes["mirrormirror"]

    assert lv.type == "mirror"

    fh = lv.open()
    for i in range(1, 513):
        assert fh.read(4096) == i.to_bytes(2, "little") * 2048


def test_lvm_multiple_data_areas() -> None:
    """Test if LVM2 raises an Error if more than 1 data area descriptor is specified.

    This test can be removed once lvm2 allows for multiple area descriptors
    """
    lvm_data = [
        c_lvm.label_header(id=b"LABELONE", offset=len(c_lvm.label_header)),
        c_lvm.pv_header(),
        c_lvm.disk_locn(offset=1),
        c_lvm.disk_locn(offset=2),
        c_lvm.disk_locn(),
    ]
    data = io.BytesIO(b"".join(struct.dumps() for struct in lvm_data))
    with pytest.raises(RuntimeError, match="There should be 1 data area descriptor, found 2"):
        LVM2(data)


def test_lvm_sizes_mismatch(lvm_inconsistent_sizes: BinaryIO) -> None:
    """Test if opening the LVM2Device uses PhysicalVolume.size when LVM2Device.size < PhysicalVolume.size."""
    lvm = LVM2(lvm_inconsistent_sizes)

    lv = lvm.vg.logical_volumes["lv"]

    # Size manually adjusted to be smaller than the actual data
    # This is the offset the data of a text file starts
    assert next(iter(lvm.devices.values())).size == 0x300000
    physical_volume = lvm.vg.pv[0]
    assert physical_volume.size == 0x8000000

    fh = lv.open()
    # Size of the stripe
    assert fh.size == 0x400000

    segment_stream = fh._runs[0][2]
    pv_stream = segment_stream._runs[0][2]
    # Check whether the underlying pv stream size is correct (The non adjusted size)
    assert pv_stream.size == 0x8000000

    fh.seek(0x3B8800)

    expected_data = b"A small file at the end of the disk"
    assert fh.read(len(expected_data)) == expected_data


def test_lvm_sizes_mismatch_missing_dev_size(lvm_inconsistent_sizes: BinaryIO) -> None:
    """Test if opening the LVM2Device uses the calculated disk size when PhysicalVolume.dev_size is missing."""
    lvm = LVM2(lvm_inconsistent_sizes)

    lv = lvm.vg.logical_volumes["lv"]

    # Size manually adjusted to be smaller than the actual data
    # This is the offset the data of a text file starts
    assert next(iter(lvm.devices.values())).size == 0x300000
    physical_volume = lvm.vg.pv[0]
    # physical_volume.size will calculate size based on the the data offset
    # and size of the stripes if dev_size is not defined
    physical_volume.dev_size = None
    assert physical_volume.size == 0x500000

    fh = lv.open()
    # Size of the stripe
    assert fh.size == 0x400000

    segment_stream = fh._runs[0][2]
    pv_stream = segment_stream._runs[0][2]
    assert pv_stream.size == 0x500000

    fh.seek(0x3B8800)

    expected_data = b"A small file at the end of the disk"
    assert fh.read(len(expected_data)) == expected_data


PV0_UUID = "pv0" + "0" * 28 + "a"
PV1_UUID = "pv1" + "0" * 28 + "b"

EXTENT = 8 * SECTOR_SIZE

METADATA = """
vg_map {
id = "vg0"
seqno = 1
format = "lvm2"
status = ["RESIZEABLE", "READ", "WRITE"]
flags = []
extent_size = 8
max_lv = 0
max_pv = 0
metadata_copies = 0

physical_volumes {

pv0 {
id = "%(pv0)s"
device = "/dev/loop0"
status = ["ALLOCATABLE"]
flags = []
dev_size = 32768
pe_start = 64
pe_count = 64
}

pv1 {
id = "%(pv1)s"
device = "/dev/loop1"
status = ["ALLOCATABLE"]
flags = []
dev_size = 32768
pe_start = 128
pe_count = 64
}
}

logical_volumes {

lv_test {
id = "lv0"
status = ["READ", "WRITE", "VISIBLE"]
flags = []
segment_count = 3

segment1 {
start_extent = 0
extent_count = 4
type = "linear"
stripe_count = 1
stripes = [
"pv0", 5
]
}

segment2 {
start_extent = 4
extent_count = 8
type = "striped"
stripe_count = 2
stripe_size = 2
stripes = [
"pv0", 16,
"pv1", 3
]
}

segment3 {
start_extent = 12
extent_count = 4
type = "striped"
stripe_count = 2
stripe_size = 4
stripes = [
"pv0", 32,
"pv1", 20
]
}
}

lv_mixed {
id = "lv1"
status = ["READ", "WRITE", "VISIBLE"]
flags = []
segment_count = 2

segment1 {
start_extent = 0
extent_count = 2
type = "linear"
stripe_count = 1
stripes = [
"pv1", 40
]
}

segment2 {
start_extent = 2
extent_count = 2
type = "zero"
}
}
}
}

contents = "Text Format Volume Group"
version = 1
""" % {"pv0": PV0_UUID, "pv1": PV1_UUID}


def _make_lvm_image(pv_uuid: str, pe_start: int, pe_count: int, tag: int) -> io.BytesIO:
    """Build a synthetic PV image with a non-zero pe_start and a position-dependent data pattern."""
    data_size = pe_count * 8 * SECTOR_SIZE

    metadata = METADATA.encode()
    mda_offset = 4096
    raw_offset = len(c_lvm.mda_header) + 2 * len(c_lvm.raw_locn)
    mda_size = raw_offset + len(metadata) + 1

    buf = io.BytesIO(bytes(pe_start * SECTOR_SIZE + data_size))

    label = c_lvm.label_header(id=b"LABELONE", sector=0, crc=0, offset=len(c_lvm.label_header), type=b"LVM2 001")
    buf.write(label.dumps())
    buf.write(c_lvm.pv_header(pv_uuid=pv_uuid.encode(), device_size=data_size).dumps())
    buf.write(c_lvm.disk_locn(offset=pe_start * SECTOR_SIZE, size=data_size).dumps())
    buf.write(c_lvm.disk_locn().dumps())
    buf.write(c_lvm.disk_locn(offset=mda_offset, size=mda_size).dumps())
    buf.write(c_lvm.disk_locn().dumps())

    buf.seek(mda_offset)
    buf.write(c_lvm.mda_header(checksum=0, magic=b"\x00" * 16, version=1, start=mda_offset, size=mda_size).dumps())
    buf.write(c_lvm.raw_locn(offset=raw_offset, size=len(metadata) + 1, checksum=0, flags=0).dumps())
    buf.write(c_lvm.raw_locn().dumps())
    buf.write(metadata + b"\x00")

    buf.seek(pe_start * SECTOR_SIZE)
    buf.write(b"".join(((tag << 28) | i).to_bytes(4, "little") for i in range(0, data_size, 4)))

    buf.seek(0)
    return buf


def _make_lvm() -> LVM2:
    return LVM2([_make_lvm_image(PV0_UUID, 64, 64, 1), _make_lvm_image(PV1_UUID, 128, 64, 2)])


def _read_mappings(mappings: list[LVM2Mapping]) -> bytes:
    out = bytearray()
    for mapping in mappings:
        fh = mapping.pv.open()
        fh.seek(mapping.pv_offset)
        out += fh.read(mapping.size)
    return bytes(out)


def test_lvm_map() -> None:
    """Test if mappings of requests crossing segment boundaries reassemble to the same bytes as reading the LV."""
    lvm = _make_lvm()
    lv = lvm.vg.logical_volumes["lv_test"]

    fh = lv.open()
    expected = fh.read(16 * EXTENT)
    assert len(expected) == 16 * EXTENT

    # Map the full volume
    mappings = lv.map()
    assert mappings[0].offset == 0
    assert sum(mapping.size for mapping in mappings) == 16 * EXTENT
    assert all(not mapping.missing for mapping in mappings)
    assert _read_mappings(mappings) == expected

    # Request crossing the linear -> striped segment boundary
    offset, size = 3 * EXTENT + 100, 3 * EXTENT + 200
    mappings = lv.map(offset, size)
    assert mappings[0].offset == offset
    assert sum(mapping.size for mapping in mappings) == size
    assert _read_mappings(mappings) == expected[offset : offset + size]

    # Request crossing the stripe width change (segment2 -> segment3)
    offset, size = 11 * EXTENT + 512, 2 * EXTENT
    mappings = lv.map(offset, size)
    assert mappings[0].offset == offset
    assert sum(mapping.size for mapping in mappings) == size
    assert _read_mappings(mappings) == expected[offset : offset + size]


def test_lvm_map_offsets() -> None:
    """Test if the reported PV identity, physical offset and logical interval are exact."""
    lvm = _make_lvm()
    lv = lvm.vg.logical_volumes["lv_test"]
    pv0 = lvm.vg.physical_volumes["pv0"]
    pv1 = lvm.vg.physical_volumes["pv1"]

    def fields(mappings: list[LVM2Mapping]) -> list[tuple]:
        return [(m.pv, m.pv_offset, m.offset, m.size) for m in mappings]

    # Linear segment: one contiguous run on pv0 at extent 5
    assert fields(lv.map(1024, 2048)) == [(pv0, 5 * EXTENT + 1024, 1024, 2048)]

    # Striped segment with stripe_size of 2 sectors (1 KiB), starting at LV offset 4 * EXTENT
    assert fields(lv.map(4 * EXTENT, 3 * 1024)) == [
        (pv0, 16 * EXTENT, 4 * EXTENT, 1024),
        (pv1, 3 * EXTENT, 4 * EXTENT + 1024, 1024),
        (pv0, 16 * EXTENT + 1024, 4 * EXTENT + 2048, 1024),
    ]

    # Second striped segment with a different stripe width of 4 sectors (2 KiB), starting at 12 * EXTENT
    assert fields(lv.map(12 * EXTENT, 3 * 2048)) == [
        (pv0, 32 * EXTENT, 12 * EXTENT, 2048),
        (pv1, 20 * EXTENT, 12 * EXTENT + 2048, 2048),
        (pv0, 32 * EXTENT + 2048, 12 * EXTENT + 4096, 2048),
    ]


def test_lvm_map_missing_device() -> None:
    """Test if mappings on a missing physical volume are flagged separately."""
    lvm = LVM2(_make_lvm_image(PV0_UUID, 64, 64, 1))
    lv = lvm.vg.logical_volumes["lv_test"]

    mappings = lv.map(4 * EXTENT, 4 * 1024)
    assert [mapping.missing for mapping in mappings] == [False, True, False, True]
    assert {mapping.pv.name for mapping in mappings if mapping.missing} == {"pv1"}
    # The PV identity and physical offset are still reported for the missing parts
    assert mappings[1].pv_offset == 3 * EXTENT


def test_lvm_map_unsupported_segment() -> None:
    """Test if unsupported segment types are not faked as zero regions."""
    lvm = _make_lvm()
    lv = lvm.vg.logical_volumes["lv_mixed"]

    # The supported part of the LV can still be mapped
    mappings = lv.map(0, 2 * EXTENT)
    assert sum(mapping.size for mapping in mappings) == 2 * EXTENT

    # A request intersecting the zero segment must raise instead of returning a fake zero region
    with pytest.raises(NotImplementedError):
        lv.map(EXTENT, 2 * EXTENT)
