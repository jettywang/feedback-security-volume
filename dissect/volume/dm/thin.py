from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

from dissect.util.stream import AlignedStream

from dissect.volume.dm.btree import BTree
from dissect.volume.dm.c_dm import SECTOR_SIZE, c_dm
from dissect.volume.exceptions import DMError

# The committed thin-pool superblock always lives at the first metadata block.
THIN_SUPERBLOCK_LOCATION = 0


@dataclass(frozen=True, slots=True)
class ThinMapping:
    """A single provisioned block mapping as recorded in a specific transaction.

    Args:
        device_id: The thin device the mapping belongs to.
        virtual_block: The block number within the thin device.
        data_block: The block number on the pool data device.
        timestamp: The provisioning timestamp stored alongside the data block.
    """

    device_id: int
    virtual_block: int
    data_block: int
    timestamp: int


class ThinPool:
    """Interact with a device-mapper thin-provision pool.

    A pool stores one committed superblock at the start of the metadata device. When a userspace transaction is held,
    the ``held_root`` field points at a shadow copy of an older, still valid superblock. Both describe independent
    transaction views and must never be mixed: the device details and data mapping tree of a view always come from the
    same root.
    """

    def __init__(self, metadata_fh: BinaryIO, data_fh: BinaryIO):
        self.metadata_fh = metadata_fh
        self.data_fh = data_fh

        self.metadata = ThinTransaction.open_root(self.metadata_fh, THIN_SUPERBLOCK_LOCATION)
        self._transactions: dict[int, ThinTransaction] = {self.metadata.location: self.metadata}

    @property
    def held_root(self) -> int:
        """Return the raw ``held_root`` block pointer recorded by the committed superblock."""
        return self.metadata.sb.held_root

    def transactions(self) -> list[ThinTransaction]:
        """List every readable transaction view in deterministic order.

        The committed view is always returned first, followed by the held view when one is present. Opening the held
        view validates it eagerly; a held root that cannot be read raises a :class:`DMError` instead of being silently
        skipped or replaced by the current view.
        """
        views = [self.metadata]
        if self.held_root not in (0, THIN_SUPERBLOCK_LOCATION):
            views.append(self.held_transaction())
        return views

    def held_transaction(self) -> ThinTransaction:
        """Open the read-only transaction view rooted at ``held_root``.

        Raises:
            DMError: If no held root is recorded or the held superblock cannot be validated.
        """
        held_root = self.held_root
        if held_root in (0, THIN_SUPERBLOCK_LOCATION):
            raise DMError("No held transaction root is recorded in the committed thin-pool superblock")

        view = self._transactions.get(held_root)
        if view is None:
            view = ThinTransaction.open_root(
                self.metadata_fh,
                held_root,
                expected_metadata_block_size=self.metadata.metadata_block_size,
                expected_data_block_size=self.metadata.data_block_size,
            )
            self._transactions[held_root] = view
        return view

    def open(
        self,
        device_id: int,
        size_hint: int | None = None,
        transaction: ThinTransaction | None = None,
    ) -> ThinDevice:
        """Open a thin device on this pool.

        No size information is stored in the pool, so it's recommended to provide a size hint.

        Args:
            device_id: The device ID to open.
            size_hint: Optional size hint for the device to open.
            transaction: Optional transaction view to read from. Defaults to the committed view. The device details
                and data mappings are always resolved from this single view, never from the current device list.
        """
        view = transaction or self.metadata
        return ThinDevice(self, view, device_id, size_hint)


class ThinTransaction:
    """A validated, read-only view on the pool metadata rooted at a single superblock.

    Use :meth:`open_root` to construct one. The constructor intentionally stays private so every view is validated
    before it can be used.
    """

    def __init__(self, fh: BinaryIO, sb: c_dm.thin_disk_superblock, location: int):
        self.fh = fh
        self.sb = sb
        self.location = location

        self.data_block_size = sb.data_block_size
        self.metadata_block_size = sb.metadata_block_size
        self.transaction_id = sb.trans_id
        self.time = sb.time

        self.data_mapping_root = sb.data_mapping_root
        self.device_details_root = sb.device_details_root

        self.data_mapping = BTree(self.fh, sb.data_mapping_root, sb.metadata_block_size)
        self.device_details = BTree(self.fh, sb.device_details_root, sb.metadata_block_size)

    @classmethod
    def open_root(
        cls,
        fh: BinaryIO,
        location: int,
        expected_metadata_block_size: int | None = None,
        expected_data_block_size: int | None = None,
    ) -> ThinTransaction:
        """Read and validate the superblock stored at ``location``.

        Args:
            fh: The metadata device, opened for reading.
            location: The metadata block number holding the superblock.
            expected_metadata_block_size: Block size recorded by the committed superblock, used to confirm the held
                view uses the same geometry.
            expected_data_block_size: Data block size recorded by the committed superblock.

        Raises:
            DMError: If the root is truncated, outside the device, has the wrong identity or incompatible geometry.
        """
        if expected_metadata_block_size is None:
            if location != THIN_SUPERBLOCK_LOCATION:
                raise DMError(
                    f"Cannot open held root block {location} without the committed metadata block size"
                )
            # The committed superblock sits at offset 0 and its fixed-size header is smaller than any valid metadata
            # block, so a header-sized read is enough to discover the block size recorded inside it.
            probe = c_dm.thin_disk_superblock(
                _read_block(fh, location, len(c_dm.thin_disk_superblock), len(c_dm.thin_disk_superblock))
            )
            block_size = probe.metadata_block_size
        else:
            block_size = expected_metadata_block_size

        block_size_bytes = block_size * SECTOR_SIZE
        block = _read_block(fh, location, block_size_bytes, len(c_dm.thin_disk_superblock))
        sb = c_dm.thin_disk_superblock(block)

        if sb.magic != c_dm.THIN_SUPERBLOCK_MAGIC:
            raise DMError(f"Invalid thin-pool superblock magic at root block {location}: {sb.magic:#x}")
        if sb.version != c_dm.THIN_VERSION:
            raise DMError(f"Unsupported thin-pool superblock version at root block {location}: {sb.version}")
        if sb.blocknr != location:
            raise DMError(
                f"Superblock identity mismatch at root block {location}: on-disk blocknr is {sb.blocknr}"
            )
        if sb.metadata_block_size <= 0 or sb.data_block_size <= 0:
            raise DMError(f"Invalid block sizes in superblock at root block {location}")
        if expected_metadata_block_size is not None and sb.metadata_block_size != expected_metadata_block_size:
            raise DMError(
                f"Metadata block size mismatch at root block {location}: "
                f"{sb.metadata_block_size} != {expected_metadata_block_size}"
            )
        if expected_data_block_size is not None and sb.data_block_size != expected_data_block_size:
            raise DMError(
                f"Data block size mismatch at root block {location}: "
                f"{sb.data_block_size} != {expected_data_block_size}"
            )

        _validate_root(fh, sb.data_mapping_root, sb, location, "data_mapping_root")
        _validate_root(fh, sb.device_details_root, sb, location, "device_details_root")

        return cls(fh, sb, location)

    def device_ids(self) -> list[int]:
        """Return the device IDs recorded in this specific transaction, sorted ascending."""
        return sorted(device_id for device_id, _ in self.device_details.walk())

    def has_device(self, device_id: int) -> bool:
        """Return whether this transaction contains device details for ``device_id``."""
        return self.device_details.lookup(device_id) is not None

    def details(self, device_id: int) -> c_dm.disk_device_details:
        """Return the device details of ``device_id`` from this transaction.

        Raises:
            DMError: If the device does not exist in this specific transaction. A missing device is a metadata gap,
                never a zero-filled device.
        """
        details_buf = self.device_details.lookup(device_id)
        if not details_buf:
            # Keep the historical wording: this is the error raised by the long standing ``ThinPool.open`` path.
            raise DMError(f"Device ID is not known in pool: {device_id} (root block {self.location})")
        return c_dm.disk_device_details(details_buf)

    def mapping(self, device_id: int, virtual_block: int) -> ThinMapping | None:
        """Resolve one virtual block of ``device_id`` within this transaction.

        Returns ``None`` for a genuine, unprovisioned hole of a device that exists. Raises :class:`DMError` when the
        device itself is absent, so a missing device or another device's mapping can never be read as zeroes of the
        target device.
        """
        if not self.has_device(device_id):
            raise DMError(f"Device ID is not known in pool: {device_id} (root block {self.location})")

        block_info = self.data_mapping.lookup([device_id, virtual_block])
        if block_info is None:
            return None

        data_block, timestamp = _unpack_block_time(int.from_bytes(block_info, "little"))
        return ThinMapping(device_id, virtual_block, data_block, timestamp)

    def mappings(self, device_id: int) -> list[ThinMapping]:
        """Return every recorded mapping of ``device_id`` in this transaction, ordered by virtual block."""
        if not self.has_device(device_id):
            raise DMError(f"Device ID is not known in pool: {device_id} (root block {self.location})")

        subtree_root = self.data_mapping.lookup(device_id)
        if subtree_root is None:
            return []

        subtree = BTree(self.fh, int.from_bytes(subtree_root, "little"), self.metadata_block_size)
        result = []
        for virtual_block, value in subtree.walk():
            data_block, timestamp = _unpack_block_time(int.from_bytes(value, "little"))
            result.append(ThinMapping(device_id, virtual_block, data_block, timestamp))
        return result


class ThinDevice(AlignedStream):
    """Implements a readable thin device/volume bound to one specific transaction.

    Args:
        pool: The pool the device belongs to.
        transaction: The transaction view supplying the device details and mappings.
        device_id: The device ID of the device.
        size_hint: Optional device size hint.
    """

    def __init__(self, pool: ThinPool, transaction: ThinTransaction, device_id: int, size_hint: int | None = None):
        self.pool = pool
        self.transaction = transaction
        self.device_id = device_id
        self.block_size = transaction.data_block_size * SECTOR_SIZE

        self.details = transaction.details(device_id)

        super().__init__(size_hint, self.block_size)

    def _read(self, offset: int, length: int) -> bytes:
        data_fh = self.pool.data_fh
        transaction = self.transaction

        result = []
        virtual_block = offset // self.block_size
        in_block_offset = offset % self.block_size
        while length > 0:
            mapping = transaction.mapping(self.device_id, virtual_block)

            read_size = min(length, self.block_size - in_block_offset)

            if mapping is None:
                # Without a size hint, stop the unbounded read at the first unprovisioned block instead of
                # synthesising zeroes forever. With a known size holes are simply zero-filled up to the size.
                if self.size is None:
                    result.append(b"\x00" * read_size)
                    break
                result.append(b"\x00" * read_size)
            else:
                data_offset = mapping.data_block * self.block_size + in_block_offset
                data_fh.seek(data_offset)
                chunk = data_fh.read(read_size)
                if len(chunk) != read_size:
                    raise DMError(
                        f"Truncated pool data device while reading device {self.device_id} "
                        f"virtual block {virtual_block} from transaction at root block {transaction.location}"
                    )
                result.append(chunk)

            length -= read_size
            virtual_block += 1
            in_block_offset = 0

        return b"".join(result)


def _read_block(fh: BinaryIO, location: int, block_size_bytes: int, minimum_size: int) -> bytes:
    """Read up to one metadata block at ``location``.

    ``block_size_bytes`` always drives the block offset so non-zero roots are located correctly. The returned buffer
    may be shorter than a full block (for example when probing the fixed-size superblock header), but it always
    contains at least ``minimum_size`` bytes or a truncation error is raised.
    """
    if location < 0:
        raise DMError(f"Invalid negative superblock location: {location}")

    block_offset = location * block_size_bytes
    fh.seek(0, 2)
    device_size = fh.tell()
    if block_offset >= device_size:
        raise DMError(
            f"Superblock root block {location} is outside the metadata device "
            f"(offset {block_offset} >= device size {device_size})"
        )

    fh.seek(block_offset)
    buf = fh.read(block_size_bytes)
    if len(buf) < minimum_size:
        raise DMError(
            f"Truncated superblock at root block {location}: need {minimum_size} bytes, got {len(buf)}"
        )
    return buf


def _validate_root(
    fh: BinaryIO,
    root: int,
    sb: c_dm.thin_disk_superblock,
    location: int,
    field_name: str,
) -> None:
    block_size_bytes = sb.metadata_block_size * SECTOR_SIZE
    if root >= sb.metadata_nr_blocks:
        raise DMError(
            f"{field_name} {root} of superblock at root block {location} references a block beyond "
            f"metadata_nr_blocks {sb.metadata_nr_blocks}"
        )

    fh.seek(0, 2)
    device_size = fh.tell()
    block_offset = root * block_size_bytes
    if block_offset >= device_size:
        raise DMError(
            f"{field_name} {root} of superblock at root block {location} cannot be reached: "
            f"metadata device truncated at {device_size} bytes"
        )


def _unpack_block_time(block_time: int) -> tuple[int, int]:
    """Unpack a block number and timestamp from a combined block time value.

    Block numbers in device-mapper thin-provisioning store the block number in the upper 40 bits and
    a timestamp in the lower 24 bits.
    """
    block = block_time >> 24
    time = block_time & ((1 << 24) - 1)
    return block, time


# Backwards compatible alias: the former Metadata class now models a single transaction view.
Metadata = ThinTransaction
