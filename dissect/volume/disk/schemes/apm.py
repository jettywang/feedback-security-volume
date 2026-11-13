from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

from dissect.cstruct import cstruct

from dissect.volume.disk.partition import Partition
from dissect.volume.exceptions import DiskError

if TYPE_CHECKING:
    from collections.abc import Iterator

apm_def = """
/* Driver Descriptor Record, stored in block 0 of an Apple-partitioned disk.
   It is always one block in size and big endian. */
struct driver_descriptor_record {
    char    signature[2];          /* "ER" */
    uint16  block_size;            /* logical block size in bytes */
    uint32  block_count;           /* number of blocks on the disk */
    uint16  reserved_1;
    uint16  driver_count;
    uint16  driver_entry_start;
    uint16  additional_driver_count;
    uint32  driver_entry_size;
    char    reserved_2[492];
};

/* A single partition map entry. The on-disk structure is always 512 bytes,
   regardless of the disk's logical block size; on larger blocks the
   remaining (block_size - 512) bytes are padding. */
struct partition_entry {
    char    signature[2];
    uint16  reserved_1;
    uint32  partition_count;
    uint32  partition_start;
    uint32  partition_size;
    char    partition_name[32];
    char    partition_type[32];
    uint32  data_start;
    uint32  data_size;
    uint32  partition_status;
    uint32  boot_code_start;
    uint32  boot_code_size;
    uint32  boot_loader_address;
    uint32  reserved_2;
    uint32  boot_code_entry;
    uint32  reserved_3;
    uint32  boot_code_checksum;
    char    processor_type[16];
    char    reserved_4[376];
};
"""

c_apm = cstruct(endian=">").load(apm_def)


class APM:
    """Apple Partition Map."""

    def __init__(self, fh: BinaryIO, sector_size: int | None = None):
        self.fh = fh
        self.sector_size = sector_size
        self.ddr = None

        self.fh.seek(0)
        first_block = c_apm.driver_descriptor_record(self.fh)
        if first_block.signature == b"ER":
            # The Driver Descriptor Record authoritatively declares the block size.
            self.ddr = first_block
            if self.ddr.block_size <= 0:
                raise DiskError(f"Invalid APM block size in driver descriptor record: {self.ddr.block_size}.")
            if self.sector_size is None:
                # No caller-provided block size: derive it from the descriptor record.
                self.sector_size = self.ddr.block_size
            elif self.sector_size != self.ddr.block_size:
                # Never silently mix two coordinate systems: a conflict is an explicit error.
                raise DiskError(
                    f"Sector size mismatch: caller provided {self.sector_size} bytes but the driver "
                    f"descriptor record declares {self.ddr.block_size} bytes."
                )
            # The partition map starts in the block following the descriptor record.
            self.fh.seek(self.sector_size)
        else:
            # No driver descriptor record: keep the caller-provided block size, or fall back to the
            # 512-byte default, and treat the disk as starting directly with the first map entry.
            if self.sector_size is None:
                self.sector_size = 512
            self.fh.seek(0)

        self.apm = c_apm.partition_entry(self.fh)
        if self.apm.signature != b"PM":
            raise DiskError(f"Invalid APM signature, expected 'PM', got {self.apm.signature!r}.")

        self._partitions_offset = self.fh.tell() - c_apm.partition_entry.size
        self.partitions: list[Partition] = list(self._partitions())

    def _partitions(self) -> Iterator[Partition]:
        for i in range(self.apm.partition_count):
            # Entries are spaced one logical block apart, not back to back: the on-disk entry is
            # always 512 bytes, while the stride is the disk's logical block size.
            self.fh.seek(self._partitions_offset + (i * self.sector_size))
            p = c_apm.partition_entry(self.fh)

            # A truncated block yields a partial entry instead of a seemingly valid partition.
            if p.signature != b"PM":
                raise DiskError(f"Invalid APM partition entry {i + 1}, expected 'PM', got {p.signature!r}.")

            # partition_start and partition_size are expressed in logical blocks, convert to bytes.
            yield Partition(
                disk=self,
                number=i + 1,  # partitions are 1-indexed
                offset=p.partition_start * self.sector_size,
                size=p.partition_size * self.sector_size,
                type=p.partition_type.rstrip(b"\x00").decode(),
                name=p.partition_name.rstrip(b"\x00").decode(),
                raw=p,
            )
