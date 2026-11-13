# This file is still a WIP
from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING, BinaryIO, NamedTuple

from dissect.cstruct import cstruct
from dissect.util.stream import AlignedStream

from dissect.volume.exceptions import Error

if TYPE_CHECKING:
    from dissect.cstruct import Structure

log = logging.getLogger(__name__)
log.setLevel(os.getenv("DISSECT_LOG_VSS", "CRITICAL"))

vss_def = """
flag _VSS_VOLUME_SNAPSHOT_ATTRIBUTES : uint32 {
    VSS_VOLSNAP_ATTR_PERSISTENT             = 0x00000001,
    VSS_VOLSNAP_ATTR_NO_AUTORECOVERY        = 0x00000002,
    VSS_VOLSNAP_ATTR_CLIENT_ACCESSIBLE      = 0x00000004,
    VSS_VOLSNAP_ATTR_NO_AUTO_RELEASE        = 0x00000008,
    VSS_VOLSNAP_ATTR_NO_WRITERS             = 0x00000010,
    VSS_VOLSNAP_ATTR_TRANSPORTABLE          = 0x00000020,
    VSS_VOLSNAP_ATTR_NOT_SURFACED           = 0x00000040,
    VSS_VOLSNAP_ATTR_NOT_TRANSACTED         = 0x00000080,
    VSS_VOLSNAP_ATTR_HARDWARE_ASSISTED      = 0x00010000,
    VSS_VOLSNAP_ATTR_DIFFERENTIAL           = 0x00020000,
    VSS_VOLSNAP_ATTR_PLEX                   = 0x00040000,
    VSS_VOLSNAP_ATTR_IMPORTED               = 0x00080000,
    VSS_VOLSNAP_ATTR_EXPOSED_LOCALLY        = 0x00100000,
    VSS_VOLSNAP_ATTR_EXPOSED_REMOTELY       = 0x00200000,
    VSS_VOLSNAP_ATTR_AUTORECOVER            = 0x00400000,
    VSS_VOLSNAP_ATTR_ROLLBACK_RECOVERY      = 0x00800000,
    VSS_VOLSNAP_ATTR_DELAYED_POSTSNAPSHOT   = 0x01000000,
    VSS_VOLSNAP_ATTR_TXF_RECOVERY           = 0x02000000
};

enum RECORD_TYPE : uint32 {
    VOLUME_HEADER       = 0x1,
    CATALOG             = 0x2,
    STORE_INDEX         = 0x3,
    STORE_HEADER        = 0x4,
    STORE_BLOCK_RANGE   = 0x5,
    STORE_BITMAP        = 0x6
};

flag BLOCK_FLAG : uint32 {
    IS_FORWARDER    = 0x1,
    IS_OVERLAY      = 0x2,
    NOT_USED        = 0x4
};

struct volume_header {
    char    identifier[16];
    uint32  version;
    uint32  record_type;
    uint64  current_offset;
    uint64  unk0;
    uint64  unk1;
    uint64  catalog_offset;
    uint64  maximum_size;
    char    volume_identifier[16];
    char    store_volume_identifier[16];
    uint32  unk2;
    char    unk3[412];
};

struct catalog_header {
    char    identifier[16];
    uint32  version;
    uint32  record_type;
    uint64  relative_offset;
    uint64  offset;
    uint64  next_offset;
    char    unk0[80];
};

struct catalog_entry_1 {
    uint64  entry_type;
    char    unk0[120];
};

struct catalog_entry_2 {
    uint64  entry_type;
    uint64  volume_size;
    char    store_identifier[16];
    uint64  unk0;
    uint64  unk1;
    uint64  creation_time;
    char    unk2[72];
};

struct catalog_entry_3 {
    uint64  entry_type;
    uint64  store_block_list_offset;
    char    store_identifier[16];
    uint64  store_header_offset;
    uint64  store_range_list_offset;
    uint64  store_bitmap_offset;
    uint64  metadata_reference;
    uint64  allocated_size;
    uint64  store_previous_bitmap_offset;
    uint64  unk0;
    char    unk1[40];
};

struct store_header {
    char        identifier[16];
    uint32      version;
    RECORD_TYPE record_type;
    uint64      relative_offset;
    uint64      offset;
    uint64      next_offset;
    uint64      size;
    char        unk0[72];
};

struct store_information {
    char    unk_identifier[16];
    char    copy_identifier[16];
    char    copy_set_identifier[16];
    uint32  type;
    uint32  provider;
    _VSS_VOLUME_SNAPSHOT_ATTRIBUTES attributes;
    uint32  unk0;
    uint16  operating_machine_len;
    wchar   operating_machine[operating_machine_len / 2];
    uint16  service_machine_len;
    wchar   service_machine[service_machine_len / 2];
};

struct block_descriptor {
    uint64  original_offset;
    uint64  relative_offset;
    uint64  store_offset;
    BLOCK_FLAG  flags;
    uint32  allocation_bitmap;
};

struct range_descriptor {
    uint64  store_offset;
    uint64  relative_offset;
    uint64  size;
};
"""

c_vss = cstruct().load(vss_def)

RECORD_TYPE = c_vss.RECORD_TYPE
BLOCK_FLAG = c_vss.BLOCK_FLAG

VSS_IDENTIFIER = b"\x6b\x87\x08\x38\x76\xc1\x48\x4e\xb7\xae\x04\x04\x6e\x6c\xc7\x52"
VOLUME_HEADER_OFFSET = 0x1E00
BLOCK_SIZE = 0x4000
SECTOR_SIZE = 512
SECTORS_PER_BLOCK = BLOCK_SIZE // SECTOR_SIZE

CATALOG_BLOCK_SIZE = 0x4000
CATALOG_ENTRY_SIZE = 128

STORE_BLOCK_SIZE = 0x4000

STORE_BLOCKLIST_ENTRY_SIZE = 32
STORE_RANGELIST_ENTRY_SIZE = 24


class VSS:
    def __init__(self, fh: BinaryIO):
        self.fh = fh

        fh.seek(VOLUME_HEADER_OFFSET)
        self.header = c_vss.volume_header(fh)
        if self.header.identifier != VSS_IDENTIFIER:
            raise Error(
                f"Not a valid VSS identifier (got {self.header.identifier.hex()}, expected {VSS_IDENTIFIER.hex()}"
            )

        if self.header.catalog_offset == 0:
            raise Error("Catalog offset is 0")

        self.catalog = Catalog(self, self.header.catalog_offset)

    def __repr__(self) -> str:
        return (
            f"<VSS volume_identifier={self.volume_identifier} store_volume_identifier={self.store_volume_identifier}>"
        )

    @property
    def volume_identifier(self) -> uuid.UUID:
        return uuid.UUID(bytes_le=self.header.volume_identifier)

    @property
    def store_volume_identifier(self) -> uuid.UUID:
        return uuid.UUID(bytes_le=self.header.store_volume_identifier)


class Catalog:
    def __init__(self, vss: VSS, offset: int):
        self.vss = vss
        self.fh = vss.fh
        self.header, data = read_block(vss.fh, offset, c_vss.catalog_header)
        buf = memoryview(data)

        store_map = defaultdict(list)

        self.entries = []
        for entry_offset in range(0, len(data), CATALOG_ENTRY_SIZE):
            entry_buf = buf[entry_offset : entry_offset + CATALOG_ENTRY_SIZE]
            entry_type = c_vss.uint64(entry_buf[:8])

            entry = None
            if entry_type == 1:
                entry = c_vss.catalog_entry_1(entry_buf)
            elif entry_type == 2:
                entry = c_vss.catalog_entry_2(entry_buf)
            elif entry_type == 3:
                entry = c_vss.catalog_entry_3(entry_buf)
            elif entry_type == 0:
                break
            else:
                raise Error("Invalid catalog entry")

            if entry_type in (2, 3):
                store_map[entry.store_identifier].append(entry)

            self.entries.append(entry)

        self.stores = []
        for store_descriptors in store_map.values():
            self.stores.append(Store(self, store_descriptors))

        self.stores = sorted(self.stores, key=lambda store: store.creation_time)
        prev_store = None
        for i, store in enumerate(self.stores):
            store.index = i
            store.previous_store = prev_store
            if prev_store:
                prev_store.next_store = store
            prev_store = store

    def __repr__(self) -> str:
        return f"<Catalog stores={len(self.stores)}>"


class SectorSource(NamedTuple):
    """Provenance of a single 512-byte sector of a recovered block.

    Attributes:
        sector: Sector index within the block (0-31).
        origin: Where the sector data came from. One of ``store-data``, ``forwarder``,
            ``overlay``, ``volume-reverse-map``, ``volume`` or ``zero-fill-bitmap``.
        store_index: Index of the store that physically provided the data, or ``None``
            when the data came from the current volume or was zero-filled.
        file_offset: Absolute file offset the sector was read from, or ``None`` for
            zero-filled sectors.
    """

    sector: int
    origin: str
    store_index: int | None
    file_offset: int | None


class BlockRead:
    """Result of resolving one volume block through the store chain.

    Carries both the recovered ``data`` and its provenance: ``decisions`` is the
    ordered decision chain that was followed through the stores, and ``sectors``
    holds a :class:`SectorSource` for every 512-byte sector, recorded at the exact
    spot where that sector was physically read (or zero-filled). Sectors that could
    not be resolved remain ``None`` and are listed in :attr:`unknown_sectors`.
    """

    def __init__(self, block: int):
        self.block = block
        self.decisions: list[str] = []
        self.sectors: list[SectorSource | None] = [None] * SECTORS_PER_BLOCK
        self.data: bytes | None = None

    def set_source(self, origin: str, store_index: int | None, file_offset: int | None) -> None:
        """Attribute every sector of the block to a single origin."""
        for sector in range(SECTORS_PER_BLOCK):
            offset = file_offset + sector * SECTOR_SIZE if file_offset is not None else None
            self.sectors[sector] = SectorSource(sector, origin, store_index, offset)

    @property
    def unknown_sectors(self) -> list[int]:
        return [sector for sector in range(SECTORS_PER_BLOCK) if self.sectors[sector] is None]

    @property
    def complete(self) -> bool:
        """Whether the block was fully recovered and every sector has a known source."""
        return self.data is not None and len(self.data) == BLOCK_SIZE and not self.unknown_sectors

    def __repr__(self) -> str:
        return f"<BlockRead block={self.block} complete={self.complete}>"


class Store:
    def __init__(self, catalog: Catalog, descriptors: list):
        self.catalog = catalog
        self.fh = catalog.fh
        self.descriptors = descriptors
        self.previous_store = None
        self.next_store = None
        self.index = None

        if len(descriptors) != 2:
            raise NotImplementedError("Only got one descriptor")

        self.has_store_data = False
        if len(descriptors) == 2:
            self.has_store_data = True

        self.desc2, self.desc3 = descriptors
        self.volume_size = self.desc2.volume_size
        self.size = self.volume_size
        self.creation_time = self.desc2.creation_time

        # Only if store data
        self.fh.seek(self.desc3.store_header_offset)
        self.header = c_vss.store_header(self.fh)

        if self.header.identifier != VSS_IDENTIFIER:
            raise Error(
                f"Not a valid VSS identifier (got {self.header.identifier.hex()}, expected {VSS_IDENTIFIER.hex()}"
            )

        self.information = c_vss.store_information(self.fh)

        self.copy_identifier = uuid.UUID(bytes_le=self.information.copy_identifier)
        self.copy_set_identifier = uuid.UUID(bytes_le=self.information.copy_set_identifier)

        self._block_list = None
        self._range_list = None
        self._bitmap = None
        self._previous_bitmap = None

    def open(self) -> StoreStream:
        return StoreStream(self)

    @cached_property
    def block_list(self) -> BlockList:
        return BlockList(self, self.desc3.store_block_list_offset)

    @cached_property
    def range_list(self) -> RangeList:
        return RangeList(self, self.desc3.store_range_list_offset)

    @cached_property
    def bitmap(self) -> StoreBitmap:
        return StoreBitmap(self, self.desc3.store_bitmap_offset)

    @cached_property
    def previous_bitmap(self) -> StoreBitmap | None:
        if not self.desc3.store_previous_bitmap_offset:
            return None
        return StoreBitmap(self, self.desc3.store_previous_bitmap_offset)

    def _read_exact(self, offset: int, length: int = BLOCK_SIZE) -> bytes:
        self.fh.seek(offset)
        buf = self.fh.read(length)
        if len(buf) != length:
            raise Error(
                f"Short read at offset 0x{offset:x} in store {self.index} (got {len(buf)}, expected {length})"
            )
        return buf

    def read_block(self, block: int, active_store: Store | None = None) -> bytes:
        return self.read_block_ex(block, active_store=active_store).data

    def read_block_ex(
        self, block: int, active_store: Store | None = None, _visited: set[tuple[int, int]] | None = None
    ) -> BlockRead:
        """Resolve a volume block to its historical contents, recording provenance.

        Decision chain, evaluated in order:

        1. Block list hit, ``IS_FORWARDER`` with a newer store: recurse into
           ``next_store`` at ``relative_offset`` (cycle-guarded).
        2. Block list hit, ``IS_FORWARDER`` without a newer store: read
           ``relative_offset`` from this store's own file (``forwarder``).
        3. Block list hit, regular descriptor: read ``store_offset`` (``store-data``).
        4. No descriptor (or overlay-only descriptor) with a newer store: recurse
           into ``next_store`` at the same block number.
        5. Newest store, reverse mapping hit: read the block from the current
           volume (``volume-reverse-map``).
        6. Newest store, block in use by the store bitmap (and the previous
           bitmap, if any): the block was never written, return zeroes
           (``zero-fill-bitmap``).
        7. Otherwise: read the block from the current volume (``volume``).

        Afterwards, if this store is the active store and the descriptor carries
        overlay data, the bitmap-selected 512-byte sectors are patched on top of
        the forwarded/resolved block (``overlay``).

        Raises:
            Error: On circular block references or short reads, neither of which
                may be mistaken for a complete historical block.
        """
        active_store = active_store or self

        visited = _visited if _visited is not None else set()
        token = (id(self), block)
        if token in visited:
            raise Error(f"Circular block reference while resolving block {block} in store {self.index}")
        visited.add(token)

        result = BlockRead(block)
        descriptor = self.block_list.map.map.get(block)

        if descriptor and descriptor.is_forwarder:
            if self.next_store:
                next_block = descriptor.relative_offset // BLOCK_SIZE
                result = self.next_store.read_block_ex(next_block, self, visited)
                result.decisions.insert(
                    0,
                    f"store {self.index}: block {block} is a forwarder to block {next_block}"
                    f" in store {self.next_store.index}",
                )
            else:
                result.data = self._read_exact(descriptor.relative_offset)
                result.set_source("forwarder", self.index, descriptor.relative_offset)
                result.decisions.append(
                    f"store {self.index}: block {block} is a forwarder, read from own file"
                    f" at 0x{descriptor.relative_offset:x}"
                )
        elif descriptor and not descriptor.is_overlay:
            result.data = self._read_exact(descriptor.store_offset)
            result.set_source("store-data", self.index, descriptor.store_offset)
            result.decisions.append(
                f"store {self.index}: block {block} read from store data at 0x{descriptor.store_offset:x}"
            )
        elif self.next_store:
            result = self.next_store.read_block_ex(block, self, visited)
            result.decisions.insert(
                0, f"store {self.index}: no local data for block {block}, forwarding to store {self.next_store.index}"
            )
        else:
            reverse_descriptor = self.block_list.map.reverse.get(block)
            if reverse_descriptor:
                result.data = self._read_exact(block * BLOCK_SIZE)
                result.set_source("volume-reverse-map", None, block * BLOCK_SIZE)
                result.decisions.append(
                    f"store {self.index}: block {block} has a reverse mapping, read from current volume"
                    f" at 0x{block * BLOCK_SIZE:x}"
                )
            elif self.bitmap.in_use(block) and (not self.previous_bitmap or self.previous_bitmap.in_use(block)):
                result.data = b"\x00" * BLOCK_SIZE
                result.set_source("zero-fill-bitmap", None, None)
                result.decisions.append(
                    f"store {self.index}: block {block} marked unwritten in the store bitmap, zero-filled"
                )
            else:
                result.data = self._read_exact(block * BLOCK_SIZE)
                result.set_source("volume", None, block * BLOCK_SIZE)
                result.decisions.append(
                    f"store {self.index}: block {block} not found in any store, read from current volume"
                    f" at 0x{block * BLOCK_SIZE:x}"
                )

        if descriptor and active_store is self and (descriptor.is_overlay or descriptor.overlay):
            overlay = descriptor if descriptor.is_overlay else descriptor.overlay
            overlay_offset = overlay.store_offset
            bitmap = overlay.bitmap

            buf = bytearray(result.data)
            overlaid = []
            for i in range(SECTORS_PER_BLOCK):
                if (bitmap >> i) & 0x00000001:
                    sector_offset = overlay_offset + (i * SECTOR_SIZE)
                    buf[i * SECTOR_SIZE : (i + 1) * SECTOR_SIZE] = self._read_exact(sector_offset, SECTOR_SIZE)
                    result.sectors[i] = SectorSource(i, "overlay", self.index, sector_offset)
                    overlaid.append(i)
            result.data = bytes(buf)
            result.decisions.append(f"store {self.index}: applied overlay to sectors {overlaid}")

        return result


class StoreStream(AlignedStream):
    def __init__(self, store: Store):
        self.store = store
        super().__init__(size=store.size, align=BLOCK_SIZE)

    def _read(self, offset: int, length: int) -> bytes:
        r = []
        blockidx = offset // BLOCK_SIZE

        while length > 0:
            r.append(self.store.read_block(blockidx))
            blockidx += 1
            length -= BLOCK_SIZE

        return b"".join(r)


class BlockList:
    def __init__(self, store: Store, offset: int):
        self.store = store
        self.offset = offset

        # t = time.time()
        self.header, data = read_block(store.fh, offset, c_vss.store_header)
        # print time.time() - t

        if self.header.record_type != RECORD_TYPE.STORE_INDEX:
            raise Error(
                "invalid store header type for block list"
                f" (got 0x{self.header.record_type:x}, expected 0x{RECORD_TYPE.STORE_INDEX:x}"
            )

        # t = time.time()
        buf = memoryview(data)
        self.map = BlockMap()
        for entry_offset in range(0, len(data), STORE_BLOCKLIST_ENTRY_SIZE):
            entry_buf = buf[entry_offset : entry_offset + STORE_BLOCKLIST_ENTRY_SIZE]
            if entry_buf == b"\x00" * STORE_BLOCKLIST_ENTRY_SIZE:
                break
            entry = BlockDescriptor(entry_buf)
            # self.entries.append(entry)
            self.map.add(entry)
        # print time.time() - t


class RangeList:
    def __init__(self, store: Store, offset: int):
        self.store = store
        self.offset = offset

        self.header, data = read_block(store.fh, offset, c_vss.store_header)

        if self.header.record_type != RECORD_TYPE.STORE_BLOCK_RANGE:
            raise Error(
                "invalid store header type for block list"
                f" (got 0x{self.header.record_type:x}, expected 0x{RECORD_TYPE.STORE_BLOCK_RANGE:x}"
            )

        buf = memoryview(data)
        self.entries = []
        for entry_offset in range(0, len(data), STORE_RANGELIST_ENTRY_SIZE):
            entry_buf = buf[entry_offset : entry_offset + STORE_RANGELIST_ENTRY_SIZE]
            if entry_buf == b"\x00" * STORE_RANGELIST_ENTRY_SIZE:
                break
            entry = c_vss.range_descriptor(entry_buf)
            self.entries.append(entry)


class StoreBitmap:
    def __init__(self, store: Store, offset: int):
        self.store = store
        self.offset = offset

        self.header, self.data = read_block(store.fh, offset, c_vss.store_header)

        if self.header.record_type != RECORD_TYPE.STORE_BITMAP:
            raise Error(
                "invalid store header type for bitmap"
                f" (got 0x{self.header.record_type:x}, expected 0x{RECORD_TYPE.STORE_BITMAP:x}"
            )

        buf = memoryview(self.data)
        self.test = []
        for i in range(0, len(self.data) // 4, 4):
            b = buf[i : i + 4]
            val = c_vss.uint32(b)
            for _ in range(32):
                if val & 0x00000001 == 0:
                    self.test.append(False)
                else:
                    self.test.append(True)
                val >>= 1

    def has_offset(self, offset: int) -> bool:
        return self.in_use(offset // BLOCK_SIZE)

    def in_use(self, block: int) -> bool:
        return not self.is_set(block)

    def is_set(self, block: int) -> bool:
        return (self.data[block // 8] & (1 << (block % 8))) != 0

    def __getitem__(self, block: int) -> bool:
        return self.in_use(block)


class BlockMap:
    def __init__(self):
        self.map = {}
        self.reverse = {}

    def add(self, descriptor: BlockDescriptor) -> None:
        if not descriptor.is_used:
            return

        blockmap = self.map
        reversemap = self.reverse
        key = descriptor.original_offset // BLOCK_SIZE

        if not descriptor.is_overlay:
            revkey = descriptor.relative_offset // BLOCK_SIZE
            try:
                revexist = reversemap[revkey]
            except KeyError:
                revexist = None

            if revexist:
                descriptor.original_offset = revexist.relative_offset
                del reversemap[revkey]

        if descriptor.is_forwarder and descriptor.original_offset == descriptor.relative_offset:
            return

        try:
            existing = blockmap[key]
        except KeyError:
            existing = None

        if existing:
            if descriptor.is_overlay:
                overlay = existing if existing.is_overlay else existing.overlay

                if overlay:
                    overlay.bitmap |= descriptor.bitmap
                else:
                    existing.overlay = descriptor
                return

            if existing.is_overlay:
                descriptor.overlay = existing
            else:
                descriptor.overlay = existing.overlay

        blockmap[key] = descriptor

        if descriptor.is_forwarder:
            revkey = descriptor.relative_offset // BLOCK_SIZE
            reversemap[revkey] = descriptor

    def get_descriptor(self, offset: int) -> None:
        pass

    def __getitem__(self, block: int) -> BlockMap:
        return self


class BlockDescriptor:
    __slots__ = (
        "bitmap",
        "flags",
        "is_forwarder",
        "is_overlay",
        "is_used",
        "original_offset",
        "overlay",
        "relative_offset",
        "store",
        "store_offset",
    )

    def __init__(self, buf: bytes):
        # self.store = store

        entry = c_vss.block_descriptor(buf)
        self.original_offset = entry.original_offset
        self.relative_offset = entry.relative_offset
        self.store_offset = entry.store_offset
        self.flags = entry.flags
        self.bitmap = entry.allocation_bitmap

        self.overlay = None
        self.is_used = not bool(self.flags & BLOCK_FLAG.NOT_USED)
        self.is_overlay = bool(self.flags & BLOCK_FLAG.IS_OVERLAY)
        self.is_forwarder = bool(self.flags & BLOCK_FLAG.IS_FORWARDER)

    def __eq__(self, other: object) -> bool:
        if not isinstance(BlockDescriptor, other):
            return False

        return (
            self.original_offset == other.original_offset
            and self.relative_offset == other.relative_offset
            and self.store_offset == other.store_offset
            and self.flags == other.flags
            and self.bitmap == other.bitmap
        )

    def __repr__(self) -> str:
        return (
            f"<BlockDescriptor original_offset=0x{self.original_offset:08x}"
            f" relative_offset=0x{self.relative_offset:08x} store_offset=0x{self.store_offset:08x}"
            f" flags={self.flags} bitmap=0x{self.bitmap:08x}>"
        )


# libvshadow_store_descriptor_read_buffer


def read_block(fh: BinaryIO, offset: int, struct: type[Structure]) -> tuple[Structure, bytes]:
    header, buf = read_block_data(fh, offset, struct)

    r = [buf]
    offset = header.next_offset
    while offset != 0:
        nheader, buf = read_block_data(fh, offset, struct)
        offset = nheader.next_offset
        r.append(buf)

    return header, b"".join(r)


def read_block_data(fh: BinaryIO, offset: int, struct: type[Structure]) -> tuple[Structure, bytes]:
    fh.seek(offset)
    buf = fh.read(BLOCK_SIZE)
    header = struct(buf)

    if header.identifier != VSS_IDENTIFIER:
        raise Error(f"not a valid VSS identifier (got {header.identifier.hex()}, expected {VSS_IDENTIFIER.hex()}")

    return header, buf[128:]
