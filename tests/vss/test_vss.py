from __future__ import annotations

import io
import struct

import pytest

from dissect.volume.exceptions import Error
from dissect.volume.vss import (
    BLOCK_SIZE,
    SECTOR_SIZE,
    VSS,
    VSS_IDENTIFIER,
    SectorSource,
)

VOLUME_SIZE = 0x100000

# Layout of the synthetic diff file: three stores (oldest to newest)
STORE0 = {"header": 0x8000, "blocklist": 0xC000, "rangelist": 0x10000, "bitmap": 0x14000}
STORE1 = {"header": 0x20000, "blocklist": 0x24000, "rangelist": 0x28000, "bitmap": 0x2C000}
STORE2 = {"header": 0x34000, "blocklist": 0x38000, "rangelist": 0x3C000, "bitmap": 0x40000}

STORE0_BLOCK6_OFFSET = 0x18000  # store-data for store 0, block 6
STORE0_OVERLAY_OFFSET = 0x1C000  # overlay sectors for store 0, block 5
STORE1_BLOCK7_OFFSET = 0x30000  # store-data for store 1, block 7
VOLUME_BLOCK20_OFFSET = 0x20 * BLOCK_SIZE  # 0x80000, reached through the reverse mapping
VOLUME_BLOCK22_OFFSET = 0x22 * BLOCK_SIZE  # 0x88000, plain current-volume read

IMAGE_SIZE = 0x8C000

IS_FORWARDER = 0x1
IS_OVERLAY = 0x2

STORE0_BLOCK6 = b"A" * BLOCK_SIZE
STORE1_BLOCK7 = bytes(range(256)) * (BLOCK_SIZE // 256)
OVERLAY_SECTOR0 = b"O" * SECTOR_SIZE
OVERLAY_SECTOR3 = b"Y" * SECTOR_SIZE
VOLUME_BLOCK20 = b"V" * BLOCK_SIZE
VOLUME_BLOCK22 = b"W" * BLOCK_SIZE


def _store_header(record_type: int, offset: int) -> bytes:
    return struct.pack("<16sIIQQQQ72s", VSS_IDENTIFIER, 1, record_type, offset, offset, 0, 0, b"")


def _block_list(offset: int, entries: list[tuple[int, int, int, int, int]]) -> bytes:
    data = _store_header(3, offset)  # RECORD_TYPE.STORE_INDEX
    for original, relative, store_offset, flags, bitmap in entries:
        data += struct.pack("<QQQII", original, relative, store_offset, flags, bitmap)
    return data


def build_image(*, short: bool = False) -> bytes:
    """Build a synthetic VSS diff file with three stores.

    Store 0 (oldest):
        - block 5: forwarder to block 7 in the next store, plus an overlay
          patching sectors 0 and 3
        - block 6: regular store data
    Store 1:
        - block 7: regular store data
    Store 2 (newest):
        - block 8: forwarder to relative block 0x20, creating a reverse mapping
          so that block 0x20 resolves to the current volume
        - block 0x21: unwritten according to the store bitmap (zero-fill)
        - block 0x22: written according to the store bitmap (current volume)
    """
    size = 0x8B200 if short else IMAGE_SIZE
    block6_offset = 0x8B000 if short else STORE0_BLOCK6_OFFSET
    buf = bytearray(size)

    def put(offset: int, data: bytes) -> None:
        # Never extend the buffer: data past the image size is truncated away
        buf[offset : offset + len(data)] = data[: max(0, len(buf) - offset)]

    # Volume header, catalog at 0x4000
    put(
        0x1E00,
        struct.pack(
            "<16sIIQQQQQ16s16sI412s", VSS_IDENTIFIER, 1, 1, 0, 0, 0, 0x4000, 0, b"\x01" * 16, b"\x02" * 16, 0, b""
        ),
    )

    stores = [
        (b"\x10" * 16, 100, STORE0),
        (b"\x11" * 16, 200, STORE1),
        (b"\x12" * 16, 300, STORE2),
    ]
    entries = b""
    for store_id, creation_time, layout in stores:
        entries += struct.pack("<QQ16sQQQ72s", 2, VOLUME_SIZE, store_id, 0, 0, creation_time, b"")
        entries += struct.pack(
            "<QQ16sQQQQQQQ40s",
            3,
            layout["blocklist"],
            store_id,
            layout["header"],
            layout["rangelist"],
            layout["bitmap"],
            0,
            0,
            0,
            0,
            b"",
        )
    put(0x4000, struct.pack("<16sIIQQQ80s", VSS_IDENTIFIER, 1, 2, 0x4000, 0x4000, 0, b"") + entries)

    block_lists = {
        0: [
            (5 * BLOCK_SIZE, 7 * BLOCK_SIZE, 0, IS_FORWARDER, 0),
            (5 * BLOCK_SIZE, 0, STORE0_OVERLAY_OFFSET, IS_OVERLAY, 0b1001),
            (6 * BLOCK_SIZE, 6 * BLOCK_SIZE, block6_offset, 0, 0),
        ],
        1: [(7 * BLOCK_SIZE, 7 * BLOCK_SIZE, STORE1_BLOCK7_OFFSET, 0, 0)],
        2: [(8 * BLOCK_SIZE, 0x20 * BLOCK_SIZE, 0, IS_FORWARDER, 0)],
    }

    for index, (store_id, _, layout) in enumerate(stores):
        store_information = struct.pack("<16s16s16sIIIIHH", b"\x00" * 16, b"\x00" * 16, b"\x00" * 16, 0, 0, 0, 0, 0, 0)
        put(layout["header"], _store_header(4, layout["header"]) + store_information)
        put(layout["blocklist"], _block_list(layout["blocklist"], block_lists[index]))
        put(layout["rangelist"], _store_header(5, layout["rangelist"]))
        put(layout["bitmap"], _store_header(6, layout["bitmap"]))

    # Store 2 bitmap: block 0x22 written (bit set), block 0x21 unwritten (bit clear)
    put(STORE2["bitmap"] + 128 + 0x22 // 8, bytes([1 << (0x22 % 8)]))

    put(block6_offset, STORE0_BLOCK6)
    put(STORE1_BLOCK7_OFFSET, STORE1_BLOCK7)
    put(STORE0_OVERLAY_OFFSET + 0 * SECTOR_SIZE, OVERLAY_SECTOR0)
    put(STORE0_OVERLAY_OFFSET + 3 * SECTOR_SIZE, OVERLAY_SECTOR3)
    if not short:
        put(VOLUME_BLOCK20_OFFSET, VOLUME_BLOCK20)
        put(VOLUME_BLOCK22_OFFSET, VOLUME_BLOCK22)

    return bytes(buf)


@pytest.fixture
def vss_three_stores() -> VSS:
    return VSS(io.BytesIO(build_image()))


def test_forward_then_overlay(vss_three_stores: VSS) -> None:
    """A forwarder resolved through the next store, with overlay sectors patched on top."""
    stores = vss_three_stores.catalog.stores
    assert len(stores) == 3

    read = stores[0].read_block_ex(5)

    assert read.complete
    assert not read.unknown_sectors
    # Overlay sectors come from store 0, the rest from store 1 via the forwarder
    assert read.data[0 * SECTOR_SIZE : 1 * SECTOR_SIZE] == OVERLAY_SECTOR0
    assert read.data[3 * SECTOR_SIZE : 4 * SECTOR_SIZE] == OVERLAY_SECTOR3
    assert read.data[1 * SECTOR_SIZE : 3 * SECTOR_SIZE] == STORE1_BLOCK7[1 * SECTOR_SIZE : 3 * SECTOR_SIZE]
    assert read.data[4 * SECTOR_SIZE :] == STORE1_BLOCK7[4 * SECTOR_SIZE :]

    # Decision chain: forwarder in store 0 -> store data in store 1 -> overlay in store 0
    assert "store 0" in read.decisions[0] and "forwarder" in read.decisions[0]
    assert any("store 1" in decision and "store data" in decision for decision in read.decisions)
    assert "store 0" in read.decisions[-1] and "overlay" in read.decisions[-1]

    # Per-sector provenance matches the actual reads
    assert read.sectors[0] == SectorSource(0, "overlay", 0, STORE0_OVERLAY_OFFSET)
    assert read.sectors[3] == SectorSource(3, "overlay", 0, STORE0_OVERLAY_OFFSET + 3 * SECTOR_SIZE)
    assert read.sectors[1] == SectorSource(1, "store-data", 1, STORE1_BLOCK7_OFFSET + 1 * SECTOR_SIZE)
    assert read.sectors[31] == SectorSource(31, "store-data", 1, STORE1_BLOCK7_OFFSET + 31 * SECTOR_SIZE)

    assert stores[0].read_block(5) == read.data


def test_store_data(vss_three_stores: VSS) -> None:
    read = vss_three_stores.catalog.stores[0].read_block_ex(6)

    assert read.complete
    assert read.data == STORE0_BLOCK6
    assert read.sectors[0] == SectorSource(0, "store-data", 0, STORE0_BLOCK6_OFFSET)
    assert read.sectors[31] == SectorSource(31, "store-data", 0, STORE0_BLOCK6_OFFSET + 31 * SECTOR_SIZE)


def test_reverse_mapping_reads_volume(vss_three_stores: VSS) -> None:
    read = vss_three_stores.catalog.stores[2].read_block_ex(0x20)

    assert read.complete
    assert read.data == VOLUME_BLOCK20
    assert "reverse mapping" in read.decisions[0]
    assert read.sectors[0] == SectorSource(0, "volume-reverse-map", None, VOLUME_BLOCK20_OFFSET)
    assert read.sectors[31] == SectorSource(31, "volume-reverse-map", None, VOLUME_BLOCK20_OFFSET + 31 * SECTOR_SIZE)


def test_bitmap_zero_fill(vss_three_stores: VSS) -> None:
    read = vss_three_stores.catalog.stores[2].read_block_ex(0x21)

    assert read.complete
    assert read.data == b"\x00" * BLOCK_SIZE
    assert "zero-filled" in read.decisions[0]
    assert read.sectors[0] == SectorSource(0, "zero-fill-bitmap", None, None)
    assert read.sectors[31] == SectorSource(31, "zero-fill-bitmap", None, None)


def test_volume_read(vss_three_stores: VSS) -> None:
    read = vss_three_stores.catalog.stores[2].read_block_ex(0x22)

    assert read.complete
    assert read.data == VOLUME_BLOCK22
    assert read.sectors[0] == SectorSource(0, "volume", None, VOLUME_BLOCK22_OFFSET)
    assert read.sectors[31] == SectorSource(31, "volume", None, VOLUME_BLOCK22_OFFSET + 31 * SECTOR_SIZE)


def test_circular_reference_is_not_a_complete_block(vss_three_stores: VSS, monkeypatch: pytest.MonkeyPatch) -> None:
    """A forwarder chain that loops back must raise instead of returning garbage."""
    stores = vss_three_stores.catalog.stores
    monkeypatch.setattr(stores[2], "next_store", stores[0])

    with pytest.raises(Error, match="Circular block reference"):
        stores[2].read_block_ex(0x30)


def test_short_read_is_not_a_complete_block() -> None:
    """A truncated store read must raise instead of silently returning a partial block."""
    vss = VSS(io.BytesIO(build_image(short=True)))

    with pytest.raises(Error, match="Short read"):
        vss.catalog.stores[0].read_block_ex(6)


def test_store_stream(vss_three_stores: VSS) -> None:
    store = vss_three_stores.catalog.stores[0]
    stream = store.open()

    stream.seek(5 * BLOCK_SIZE)
    assert stream.read(BLOCK_SIZE) == store.read_block(5)
