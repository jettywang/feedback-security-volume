from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import BinaryIO

from dissect.volume.dm.c_dm import SECTOR_SIZE, c_dm

METADATA_BLOCK_SECTORS = 8
METADATA_BLOCK_SIZE = METADATA_BLOCK_SECTORS * SECTOR_SIZE
DATA_BLOCK_SECTORS = 8
DATA_BLOCK_SIZE = DATA_BLOCK_SECTORS * SECTOR_SIZE
NODE_HEADER_SIZE = len(c_dm.node_header)


@dataclass
class DeviceSpec:
    """The complete, transaction-specific state of one thin device."""

    device_id: int
    transaction_id: int
    creation_time: int
    # virtual_block -> (data_block, timestamp). Missing virtual blocks are genuine holes.
    mappings: dict[int, tuple[int, int]]


@dataclass
class TransactionSpec:
    """One independent metadata view with its own roots and transaction identity."""

    trans_id: int
    time: int
    devices: dict[int, DeviceSpec]
    held_root: int = 0


@dataclass
class _BlockAllocator:
    size: int = METADATA_BLOCK_SIZE
    offset: int = 1
    blocks: dict[int, bytes] = field(default_factory=dict)
    next_local: int = 0

    def alloc(self) -> int:
        block = self.offset + self.next_local
        self.next_local += 1
        return block

    def write(self, block: int, data: bytes) -> None:
        if len(data) != self.size:
            raise ValueError(f"Block {block} has size {len(data)}, expected {self.size}")
        self.blocks[block] = data

    @property
    def used_blocks(self) -> int:
        return self.next_local


def _encode_node(entries: list[tuple[int, bytes]], value_size: int, leaf: bool, blocknr: int) -> bytes:
    max_entries = (METADATA_BLOCK_SIZE - NODE_HEADER_SIZE) // (8 + value_size)
    if len(entries) > max_entries:
        raise ValueError("Node overflow while building synthetic thin-pool")

    flags = c_dm.node_flags.LEAF_NODE.value if leaf else c_dm.node_flags.INTERNAL_NODE.value
    header = c_dm.node_header(
        csum=0,
        flags=flags,
        blocknr=blocknr,
        nr_entries=len(entries),
        max_entries=max_entries,
        value_size=value_size,
        padding=0,
    ).dumps()

    entries = sorted(entries, key=lambda item: item[0])
    # The on-disk layout reserves full max_entries-sized key and value areas regardless of the entry count.
    key_area = b"".join(key.to_bytes(8, "little") for key, _ in entries)
    key_area += b"\x00" * (max_entries * 8 - len(key_area))
    value_area = b"".join(value for _, value in entries)
    value_area += b"\x00" * (max_entries * value_size - len(value_area))

    node = header + key_area + value_area
    return node + b"\x00" * (METADATA_BLOCK_SIZE - len(node))


def _build_mapping_tree(
    allocator: _BlockAllocator,
    devices: dict[int, DeviceSpec],
) -> int:
    # Allocate and write every per-device subtree first so the top-level values can store their real block numbers.
    subtrees: list[tuple[int, int]] = []
    for device_id in sorted(devices):
        leaf_entries = [
            (virtual_block, _pack_block_time(data_block, timestamp).to_bytes(8, "little"))
            for virtual_block, (data_block, timestamp) in sorted(devices[device_id].mappings.items())
        ]
        leaf_block = allocator.alloc()
        allocator.write(leaf_block, _encode_node(leaf_entries, value_size=8, leaf=True, blocknr=leaf_block))
        subtrees.append((device_id, leaf_block))

    top_entries = [(device_id, leaf_block.to_bytes(8, "little")) for device_id, leaf_block in subtrees]
    top_block = allocator.alloc()
    # The two-level mapping tree marks both levels as leaves; the top-level values are subtree block numbers that
    # BTree.lookup descends into by treating the value as the next root.
    allocator.write(top_block, _encode_node(top_entries, value_size=8, leaf=True, blocknr=top_block))
    return top_block


def _build_details_tree(allocator: _BlockAllocator, devices: dict[int, DeviceSpec]) -> int:
    entries: list[tuple[int, bytes]] = []
    for device_id in sorted(devices):
        spec = devices[device_id]
        details = c_dm.disk_device_details(
            mapped_blocks=len(spec.mappings),
            transaction_id=spec.transaction_id,
            creation_time=spec.creation_time,
            snapshotted_time=0,
        ).dumps()
        entries.append((device_id, details))

    block = allocator.alloc()
    allocator.write(block, _encode_node(entries, value_size=len(c_dm.disk_device_details), leaf=True, blocknr=block))
    return block


def _build_superblock(
    transaction: TransactionSpec,
    location: int,
    data_mapping_root: int,
    device_details_root: int,
    metadata_nr_blocks: int,
) -> bytes:
    sb = c_dm.thin_disk_superblock(
        csum=0,
        flags=0,
        blocknr=location,
        uuid=b"\x00" * 16,
        magic=c_dm.THIN_SUPERBLOCK_MAGIC,
        version=c_dm.THIN_VERSION,
        time=transaction.time,
        trans_id=transaction.trans_id,
        held_root=transaction.held_root,
        data_space_map_root=b"\x00" * c_dm.SPACE_MAP_ROOT_SIZE,
        metadata_space_map_root=b"\x00" * c_dm.SPACE_MAP_ROOT_SIZE,
        data_mapping_root=data_mapping_root,
        device_details_root=device_details_root,
        data_block_size=DATA_BLOCK_SECTORS,
        metadata_block_size=METADATA_BLOCK_SECTORS,
        metadata_nr_blocks=metadata_nr_blocks,
        compat_flags=0,
        compat_ro_flags=0,
        incompat_flags=0,
    ).dumps()
    return sb + b"\x00" * (METADATA_BLOCK_SIZE - len(sb))


def _pack_block_time(data_block: int, timestamp: int) -> int:
    return (data_block << 24) | timestamp


def _build_transaction_blocks(
    transaction: TransactionSpec,
    allocator: _BlockAllocator,
    superblock_block: int,
    metadata_nr_blocks: int,
) -> None:
    """Populate ``allocator`` with the btrees and superblock of one independent transaction view."""
    mapping_root = _build_mapping_tree(allocator, transaction.devices)
    details_root = _build_details_tree(allocator, transaction.devices)
    allocator.write(
        superblock_block,
        _build_superblock(transaction, superblock_block, mapping_root, details_root, metadata_nr_blocks),
    )


def build_pool(
    live: TransactionSpec,
    held: TransactionSpec | None,
    data_blocks: dict[int, bytes],
    *,
    metadata_nr_blocks: int = 64,
) -> tuple[BinaryIO, BinaryIO]:
    """Build an in-memory metadata/data pair containing independent transaction views.

    Each transaction gets a disjoint metadata block range, so its btrees never overwrite another view. The held
    transaction is written at a non-zero block and referenced from the committed superblock through ``held_root``.
    """
    metadata_image = bytearray(metadata_nr_blocks * METADATA_BLOCK_SIZE)

    def _emit(allocator: _BlockAllocator) -> None:
        for block, payload in allocator.blocks.items():
            metadata_image[block * METADATA_BLOCK_SIZE : (block + 1) * METADATA_BLOCK_SIZE] = payload

    if held is not None:
        # The held view occupies the first range after block 0; its superblock closes that range.
        held_allocator = _BlockAllocator(offset=1)
        held_mapping_root = _build_mapping_tree(held_allocator, held.devices)
        held_details_root = _build_details_tree(held_allocator, held.devices)
        held_superblock_block = held_allocator.alloc()
        held_allocator.write(
            held_superblock_block,
            _build_superblock(
                held, held_superblock_block, held_mapping_root, held_details_root, metadata_nr_blocks
            ),
        )
        live.held_root = held_superblock_block
        _emit(held_allocator)

        live_offset = 1 + held_allocator.used_blocks
    else:
        live_offset = 1

    # The committed view owns block 0 (its superblock); its btrees live in a range disjoint from the held view.
    live_allocator = _BlockAllocator(offset=live_offset)
    _build_transaction_blocks(live, live_allocator, 0, metadata_nr_blocks)

    if live_offset + live_allocator.used_blocks > metadata_nr_blocks:
        raise ValueError("Synthetic metadata pool too small for both transaction views")

    _emit(live_allocator)

    metadata = io.BytesIO(bytes(metadata_image))

    highest = max(data_blocks, default=-1)
    data_image = bytearray((highest + 1) * DATA_BLOCK_SIZE)
    for data_block, payload in data_blocks.items():
        if len(payload) != DATA_BLOCK_SIZE:
            raise ValueError(f"Data block {data_block} has size {len(payload)}, expected {DATA_BLOCK_SIZE}")
        data_image[data_block * DATA_BLOCK_SIZE : (data_block + 1) * DATA_BLOCK_SIZE] = payload
    data = io.BytesIO(bytes(data_image))

    return metadata, data


def pattern_block(marker: int) -> bytes:
    """Return a full data block filled with a recognizable repeated two-byte marker."""
    return marker.to_bytes(2, "little") * (DATA_BLOCK_SIZE // 2)
