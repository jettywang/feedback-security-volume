from __future__ import annotations

import io

import pytest

from dissect.volume.dm.btree import BTree
from dissect.volume.dm.c_dm import SECTOR_SIZE, c_dm
from dissect.volume.dm.thin import THIN_SUPERBLOCK_LOCATION, ThinPool, ThinTransaction
from dissect.volume.exceptions import DMError

from tests.dm._thin_pool import (
    DATA_BLOCK_SIZE,
    METADATA_BLOCK_SIZE,
    DeviceSpec,
    TransactionSpec,
    build_pool,
    pattern_block,
)

HELD_TRANS_ID = 100
LIVE_TRANS_ID = 200
DEVICE_ID = 0
OTHER_DEVICE_ID = 1


@pytest.fixture
def two_transaction_pool() -> tuple[ThinPool, dict[int, bytes], int]:
    """A small pool with a held and a committed view that disagree about device 0.

    Device 0 maps virtual blocks 0, 1 and 3 in both views but to different data blocks; virtual block 2 is a genuine
    hole. Device 1 exists only in the committed view and must never leak into the held view.
    """
    data_blocks = {index: pattern_block(0x1000 + index) for index in range(7)}

    held = TransactionSpec(
        trans_id=HELD_TRANS_ID,
        time=10,
        devices={
            DEVICE_ID: DeviceSpec(
                device_id=DEVICE_ID,
                transaction_id=HELD_TRANS_ID,
                creation_time=10,
                mappings={0: (0, 1), 1: (1, 1), 3: (2, 1)},
            ),
        },
    )
    live = TransactionSpec(
        trans_id=LIVE_TRANS_ID,
        time=20,
        devices={
            DEVICE_ID: DeviceSpec(
                device_id=DEVICE_ID,
                transaction_id=HELD_TRANS_ID,
                creation_time=10,
                mappings={0: (3, 2), 1: (4, 2), 3: (5, 2)},
            ),
            OTHER_DEVICE_ID: DeviceSpec(
                device_id=OTHER_DEVICE_ID,
                transaction_id=LIVE_TRANS_ID,
                creation_time=20,
                mappings={0: (6, 2)},
            ),
        },
    )

    metadata_fh, data_fh = build_pool(live, held, data_blocks)
    pool = ThinPool(metadata_fh, data_fh)
    return pool, data_blocks, live.held_root


def test_transactions_list_distinct_roots_and_identities(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, _, held_root = two_transaction_pool

    assert pool.held_root == held_root
    assert held_root != THIN_SUPERBLOCK_LOCATION

    transactions = pool.transactions()
    assert len(transactions) == 2

    committed, held = transactions
    assert committed.location == THIN_SUPERBLOCK_LOCATION
    assert committed.transaction_id == LIVE_TRANS_ID
    assert held.location == held_root
    assert held.transaction_id == HELD_TRANS_ID

    # Every view carries its own independent roots and the pool geometry.
    for transaction in transactions:
        assert transaction.data_mapping_root != held.device_details_root
        assert transaction.data_block_size * SECTOR_SIZE == DATA_BLOCK_SIZE
        assert transaction.metadata_block_size * SECTOR_SIZE == METADATA_BLOCK_SIZE

    # The two views do not share either btree root.
    live = committed
    assert held.data_mapping_root != live.data_mapping_root
    assert held.device_details_root != live.device_details_root

    # Device membership is view specific.
    assert live.device_ids() == [DEVICE_ID, OTHER_DEVICE_ID]
    assert held.device_ids() == [DEVICE_ID]


def test_same_device_reads_distinct_known_content_per_transaction(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, data_blocks, held_root = two_transaction_pool
    held = pool.held_transaction()
    live = pool.metadata

    held_device = pool.open(DEVICE_ID, DATA_BLOCK_SIZE, transaction=held)
    live_device = pool.open(DEVICE_ID, DATA_BLOCK_SIZE, transaction=live)

    assert held_device.transaction.location == held_root
    assert live_device.transaction.location == THIN_SUPERBLOCK_LOCATION

    # The same device ID resolves to different physical data blocks in each transaction.
    assert held.mapping(DEVICE_ID, 0).data_block == 0
    assert live.mapping(DEVICE_ID, 0).data_block == 3

    held_content = held_device.read(DATA_BLOCK_SIZE)
    live_content = live_device.read(DATA_BLOCK_SIZE)
    assert held_content == data_blocks[0]
    assert live_content == data_blocks[3]
    assert held_content != live_content


def test_cross_block_read_spans_data_hole_and_following_data(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, data_blocks, held_root = two_transaction_pool
    held = pool.held_transaction()
    device = pool.open(DEVICE_ID, 4 * DATA_BLOCK_SIZE, transaction=held)

    whole = device.read(4 * DATA_BLOCK_SIZE)
    assert whole[0:DATA_BLOCK_SIZE] == data_blocks[0]
    assert whole[DATA_BLOCK_SIZE : 2 * DATA_BLOCK_SIZE] == data_blocks[1]
    assert whole[2 * DATA_BLOCK_SIZE : 3 * DATA_BLOCK_SIZE] == b"\x00" * DATA_BLOCK_SIZE
    assert whole[3 * DATA_BLOCK_SIZE :] == data_blocks[2]

    # An unaligned request that crosses a block boundary still stitches the two adjacent mappings together.
    misaligned_start = DATA_BLOCK_SIZE - 96
    device.seek(misaligned_start)
    spanning = device.read(256)
    expected = data_blocks[0][misaligned_start:] + data_blocks[1][: 256 - 96]
    assert spanning == expected

    # A genuine hole of an existing device reports no mapping, but reads back as zeroes.
    assert held.mapping(DEVICE_ID, 2) is None
    assert device.readoffset(2 * DATA_BLOCK_SIZE, DATA_BLOCK_SIZE) == b"\x00" * DATA_BLOCK_SIZE
    assert held.location == held_root


def test_missing_device_and_foreign_mappings_are_not_zero_regions(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, _, _ = two_transaction_pool
    held = pool.held_transaction()

    # Device 1 only exists in the committed view. Opening or querying it in the held view is an explicit error rather
    # than an interpretation of the current device list or a zero-filled device.
    with pytest.raises(DMError, match="Device ID is not known in pool: 1"):
        pool.open(OTHER_DEVICE_ID, transaction=held)
    with pytest.raises(DMError, match="Device ID is not known in pool: 1"):
        held.mapping(OTHER_DEVICE_ID, 0)
    with pytest.raises(DMError, match="Device ID is not known in pool: 1"):
        held.details(OTHER_DEVICE_ID)
    with pytest.raises(DMError, match="Device ID is not known in pool: 1"):
        held.mappings(OTHER_DEVICE_ID)

    assert held.has_device(OTHER_DEVICE_ID) is False
    assert pool.metadata.has_device(OTHER_DEVICE_ID) is True

    # The held view still resolves device 0 through its own roots, never through device 1's committed mapping.
    held_device = pool.open(DEVICE_ID, DATA_BLOCK_SIZE, transaction=held)
    assert held_device.read(DATA_BLOCK_SIZE) == pattern_block(0x1000)


def test_details_and_mappings_come_from_selected_root_only(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, _, _ = two_transaction_pool
    held = pool.held_transaction()
    live = pool.metadata

    held_details = held.details(DEVICE_ID)
    live_details = live.details(DEVICE_ID)
    assert held_details.transaction_id == HELD_TRANS_ID
    assert held_details.mapped_blocks == 3

    held_mappings = held.mappings(DEVICE_ID)
    live_mappings = live.mappings(DEVICE_ID)
    assert [(mapping.virtual_block, mapping.data_block) for mapping in held_mappings] == [
        (0, 0),
        (1, 1),
        (3, 2),
    ]
    assert [(mapping.virtual_block, mapping.data_block) for mapping in live_mappings] == [
        (0, 3),
        (1, 4),
        (3, 5),
    ]
    assert all(mapping.device_id == DEVICE_ID for mapping in held_mappings)


def test_pool_without_held_root_reports_unavailable_instead_of_fallback(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, _, _ = two_transaction_pool
    metadata = pool.metadata_fh.getvalue()

    # Rewrite the committed superblock so it no longer records a held root.
    committed = c_dm.thin_disk_superblock(metadata)
    rebuilt = c_dm.thin_disk_superblock(committed.dumps())
    rebuilt.held_root = 0
    image = bytearray(metadata)
    image[0 : len(rebuilt.dumps())] = rebuilt.dumps()

    no_held_pool = ThinPool(io.BytesIO(bytes(image)), pool.data_fh)
    assert no_held_pool.transactions() == [no_held_pool.metadata]
    with pytest.raises(DMError, match="No held transaction root"):
        no_held_pool.held_transaction()


def test_truncated_held_superblock_is_rejected() -> None:
    data_blocks = {0: pattern_block(0x3000)}
    held = TransactionSpec(
        trans_id=HELD_TRANS_ID,
        time=10,
        devices={DEVICE_ID: DeviceSpec(DEVICE_ID, HELD_TRANS_ID, 10, {0: (0, 1)})},
    )
    live = TransactionSpec(
        trans_id=LIVE_TRANS_ID,
        time=20,
        devices={DEVICE_ID: DeviceSpec(DEVICE_ID, LIVE_TRANS_ID, 20, {0: (0, 2)})},
    )
    metadata_fh, _ = build_pool(live, held, data_blocks)
    full_image = metadata_fh.getvalue()

    committed = c_dm.thin_disk_superblock(full_image[:METADATA_BLOCK_SIZE])
    held_root = committed.held_root

    # A separate read-only device that ends partway through the held superblock. Opening the held root directly must
    # report the truncation instead of silently falling back to the committed view.
    torn_size = held_root * METADATA_BLOCK_SIZE + len(c_dm.thin_disk_superblock) - 8
    torn_fh = io.BytesIO(full_image[:torn_size])

    with pytest.raises(DMError, match="Truncated superblock"):
        ThinTransaction.open_root(
            torn_fh,
            held_root,
            expected_metadata_block_size=committed.metadata_block_size,
            expected_data_block_size=committed.data_block_size,
        )


def test_held_root_outside_device_is_rejected(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, _, _ = two_transaction_pool
    committed = c_dm.thin_disk_superblock(pool.metadata_fh.getvalue())
    rebuilt = c_dm.thin_disk_superblock(committed.dumps())
    rebuilt.held_root = committed.metadata_nr_blocks + 50

    image = bytearray(pool.metadata_fh.getvalue())
    image[0 : len(rebuilt.dumps())] = rebuilt.dumps()

    broken_pool = ThinPool(io.BytesIO(bytes(image)), pool.data_fh)
    with pytest.raises(DMError, match="outside the metadata device"):
        broken_pool.held_transaction()


def test_held_superblock_identity_mismatch_is_rejected(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, _, held_root = two_transaction_pool
    image = bytearray(pool.metadata_fh.getvalue())
    offset = held_root * METADATA_BLOCK_SIZE
    held_sb = c_dm.thin_disk_superblock(bytes(image[offset : offset + METADATA_BLOCK_SIZE]))
    held_sb.blocknr = held_root + 1
    image[offset : offset + len(held_sb.dumps())] = held_sb.dumps()

    broken_pool = ThinPool(io.BytesIO(bytes(image)), pool.data_fh)
    with pytest.raises(DMError, match="Superblock identity mismatch"):
        broken_pool.held_transaction()


def test_held_geometry_mismatch_is_rejected(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, _, held_root = two_transaction_pool
    image = bytearray(pool.metadata_fh.getvalue())
    offset = held_root * METADATA_BLOCK_SIZE
    held_sb = c_dm.thin_disk_superblock(bytes(image[offset : offset + METADATA_BLOCK_SIZE]))
    held_sb.data_block_size = held_sb.data_block_size * 2
    image[offset : offset + len(held_sb.dumps())] = held_sb.dumps()

    broken_pool = ThinPool(io.BytesIO(bytes(image)), pool.data_fh)
    with pytest.raises(DMError, match="Data block size mismatch"):
        broken_pool.held_transaction()


def test_btree_root_beyond_device_is_rejected() -> None:
    data_blocks = {0: pattern_block(0x2000)}
    live = TransactionSpec(
        trans_id=LIVE_TRANS_ID,
        time=1,
        devices={DEVICE_ID: DeviceSpec(DEVICE_ID, LIVE_TRANS_ID, 1, {0: (0, 1)})},
    )
    metadata_fh, data_fh = build_pool(live, None, data_blocks)
    image = bytearray(metadata_fh.getvalue())

    committed = c_dm.thin_disk_superblock(bytes(image[:METADATA_BLOCK_SIZE]))
    committed.data_mapping_root = committed.metadata_nr_blocks + 10
    image[0 : len(committed.dumps())] = committed.dumps()

    with pytest.raises(DMError, match="data_mapping_root .* beyond"):
        ThinPool(io.BytesIO(bytes(image)), data_fh)


def test_truncated_btree_node_surfaces_as_dm_error() -> None:
    # A root that points inside the file but whose node block is only partially present must raise a DMError rather
    # than bubbling up a low-level struct parse failure.
    metadata_block_sectors = METADATA_BLOCK_SIZE // SECTOR_SIZE
    truncated_fh = io.BytesIO(b"\x00" * (METADATA_BLOCK_SIZE + 32))
    tree = BTree(truncated_fh, root=1, block_size=metadata_block_sectors)

    with pytest.raises(DMError, match="Truncated btree node at block 1"):
        tree._read_node(1)


def test_unreachable_btree_root_is_rejected_at_open(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, _, _ = two_transaction_pool
    mapping_root = pool.metadata.data_mapping_root
    # Cut before the mapping root while leaving the committed superblock itself readable.
    truncated_fh = io.BytesIO(pool.metadata_fh.getvalue()[: mapping_root * METADATA_BLOCK_SIZE])

    with pytest.raises(DMError, match="data_mapping_root .* cannot be reached"):
        ThinPool(truncated_fh, pool.data_fh)


def test_truncated_data_device_raises_instead_of_silently_short_read(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, _, _ = two_transaction_pool
    held = pool.held_transaction()
    # Device 0's first held mapping points at data block 0; remove the tail of that block.
    short_data = io.BytesIO(pool.data_fh.getvalue()[: DATA_BLOCK_SIZE // 2])
    pool.data_fh = short_data

    device = pool.open(DEVICE_ID, DATA_BLOCK_SIZE, transaction=held)
    with pytest.raises(DMError, match="Truncated pool data device"):
        device.read(DATA_BLOCK_SIZE)


def test_default_open_keeps_committed_view_backwards_compatible(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, data_blocks, _ = two_transaction_pool
    device = pool.open(DEVICE_ID, DATA_BLOCK_SIZE)
    assert device.transaction.location == THIN_SUPERBLOCK_LOCATION
    assert device.read(DATA_BLOCK_SIZE) == data_blocks[3]


def test_existing_device_without_mappings_is_distinct_from_missing_device() -> None:
    empty_device_id = 7
    live = TransactionSpec(
        trans_id=LIVE_TRANS_ID,
        time=1,
        devices={empty_device_id: DeviceSpec(empty_device_id, LIVE_TRANS_ID, 1, {})},
    )
    metadata_fh, data_fh = build_pool(live, None, {})
    pool = ThinPool(metadata_fh, data_fh)

    # The device exists, so it enumerates and reports holes/empty mappings rather than a "not known" error.
    assert pool.metadata.device_ids() == [empty_device_id]
    assert pool.metadata.has_device(empty_device_id) is True
    assert pool.metadata.mappings(empty_device_id) == []
    assert pool.metadata.mapping(empty_device_id, 0) is None
    assert pool.metadata.details(empty_device_id).mapped_blocks == 0

    with pytest.raises(DMError, match="Device ID is not known in pool: 8"):
        pool.metadata.mapping(8, 0)


def test_committed_root_can_be_opened_without_geometry_hint(
    two_transaction_pool: tuple[ThinPool, dict[int, bytes], int],
) -> None:
    pool, _, _ = two_transaction_pool
    raw = pool.metadata_fh.getvalue()

    # No expected block sizes: the committed root at block 0 is self-describing and must bootstrap itself.
    bootstrapped = ThinTransaction.open_root(io.BytesIO(raw), THIN_SUPERBLOCK_LOCATION)
    assert bootstrapped.location == THIN_SUPERBLOCK_LOCATION
    assert bootstrapped.metadata_block_size * SECTOR_SIZE == METADATA_BLOCK_SIZE
    assert bootstrapped.device_ids() == [DEVICE_ID, OTHER_DEVICE_ID]

    # A non-zero root cannot be located without the committed geometry, so this is an explicit error, not a guess.
    with pytest.raises(DMError, match="without the committed metadata block size"):
        ThinTransaction.open_root(io.BytesIO(raw), pool.held_root)
