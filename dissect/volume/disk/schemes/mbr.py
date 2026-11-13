from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, BinaryIO

from dissect.cstruct import cstruct

from dissect.volume.disk.partition import Partition
from dissect.volume.exceptions import DiskError

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

mbr_def = """
typedef struct part_s {
    uint8   bootable;       // +0: 0x80/0x00 - bootable/not bootable
    uint8   start_head;     // +1: head (start)
    uint16  start_cyl_sec;  // +2: cyl+sect (start)
    uint8   type;           // +4: type
    uint8   end_head;       // +5: head (end)
    uint16  end_cyl_sec;    // +6: cyl+sec (end)
    uint32  sector_ofs;     // +8: offset in sectors
    uint32  sector_size;    // +12: size in sectors
} part;

typedef struct mbr_s {
    char    bootcode[0x1b8];
    uint32  vol_no;
    uint16  pad1;
    part    part[4];
    uint16  bootsig;
} mbr;
"""

c_mbr = cstruct().load(mbr_def)

EXTENDED_PARTITION_TYPES = (0x05, 0x0F, 0x85)


class MBR:
    """Master Boot Record."""

    def __init__(self, fh: BinaryIO, sector_size: int = 512):
        self.fh = fh
        self.sector_size = sector_size
        self.offset = fh.tell()
        self.mbr = c_mbr.mbr_s(fh)

        if self.mbr.bootsig != 0xAA55:
            raise DiskError(f"Invalid MBR boot signature. Expected 0xaa55, got 0x{self.mbr.bootsig:x}.")

        # This sucks but don't have a better way atm
        sig = self.mbr.bootcode[3:11]
        if any(v in sig for v in [b"MSDOS", b"MSWIN", b"NTFS", b"FAT", b"EXFAT", b"-FVE-FS-", b"SYSLINUX"]):
            raise DiskError("Sector is a filesystem VBR, not an MBR")

        if self.mbr.bootcode[18:38] == b"Hit Esc for .altboot" or self.mbr.bootcode[168:174] == b"\r\nQNX ":
            raise DiskError("Sector is a QNX Boot Sector, not an MBR")

        self.partitions: list[Partition] = list(self._partitions(self.mbr, self.offset))

    def _partitions(self, mbr: c_mbr.mbr_s, offset: int, num_start: int = 0) -> Iterator[Partition]:
        # `offset` is the absolute start of `mbr` in the stream. For the primary MBR
        # this is the start of the disk, which is not necessarily 0. Each extended
        # partition entry in the primary MBR defines its own container with its own EBR
        # chain; all partitions are numbered sequentially in the order they are found.
        number = num_start + 1

        for partition in mbr.part:
            if partition.type == 0x00:
                continue

            if partition.type in EXTENDED_PARTITION_TYPES:
                # The first EBR is at the start of the extended container. The size of
                # the extended partition entry is the size of the container.
                container_start = offset + partition.sector_ofs * self.sector_size
                container_size = partition.sector_size * self.sector_size
                number = yield from self._extended_partitions(
                    container_start=container_start,
                    first_ebr=container_start,
                    container_size=container_size,
                    number=number,
                )
                continue

            yield Partition(
                disk=self,
                number=number,
                offset=offset + partition.sector_ofs * self.sector_size,
                size=partition.sector_size * self.sector_size,
                type=partition.type,
                flags=partition.bootable,
                raw=partition,
            )
            number += 1

    def _extended_partitions(
        self, container_start: int, first_ebr: int, container_size: int, number: int
    ) -> Generator[Partition, None, int]:
        """Walk a single extended partition chain.

        ``container_start`` and ``container_size`` describe the extended partition
        container. ``first_ebr`` is the absolute address of the first EBR of the chain.
        Logical partitions are relative to the *current* EBR, while the link to the next
        EBR is relative to the *first* EBR. These are distinct address bases and must
        not be mixed. Both the current EBR and every logical partition have to stay
        within the container. Revisiting an EBR address ends the walk, so self-referencing
        or looping chains terminate without a fixed recursion limit.

        Yields the logical partitions and returns the next unused partition number.
        """
        container_end = container_start + container_size
        visited: set[int] = set()
        pending = deque([first_ebr])

        while pending:
            current_ebr = pending.popleft()

            if current_ebr in visited or not container_start <= current_ebr < container_end:
                continue
            visited.add(current_ebr)

            try:
                ebr = self._read_ebr(current_ebr)
            except (EOFError, OSError, ValueError):
                continue

            for partition in ebr.part:
                if partition.type == 0x00:
                    continue

                if partition.type in EXTENDED_PARTITION_TYPES:
                    # The next EBR is relative to the first EBR, not to the current one
                    next_ebr = first_ebr + partition.sector_ofs * self.sector_size
                    if next_ebr not in visited and container_start <= next_ebr < container_end:
                        pending.append(next_ebr)
                    continue

                # Logical partitions are relative to the current EBR
                part_offset = current_ebr + partition.sector_ofs * self.sector_size
                size = partition.sector_size * self.sector_size

                # Never surface data that falls outside the extended partition container
                if not container_start <= part_offset < container_end or part_offset + size > container_end:
                    continue

                yield Partition(
                    disk=self,
                    number=number,
                    offset=part_offset,
                    size=size,
                    type=partition.type,
                    flags=partition.bootable,
                    raw=partition,
                )
                number += 1

        return number

    def _read_ebr(self, offset: int) -> c_mbr.mbr_s:
        self.fh.seek(offset)
        return c_mbr.mbr_s(self.fh)
