from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple

from dissect.util.stream import AlignedStream, MappingStream
from dissect.volume.exceptions import RAIDError, RAIDStripeReadError

if TYPE_CHECKING:
    from dissect.volume.raid.raid import DiskMap, PhysicalDisk, VirtualDisk


class MemberEvidence(NamedTuple):
    """Forensic evidence for a single member involved in a failed stripe read.

    Args:
        disk_idx: The RAID device index of the member (its position in the virtual disk map).
        reason: Why the member could not supply its stripe chunk.
        member_offset: The byte offset on the physical member where the chunk should have been read.
        expected_length: The number of bytes expected from the member.
        actual_length: The number of bytes actually returned, or ``None`` if no read was possible.
    """

    disk_idx: int
    reason: str
    member_offset: int
    expected_length: int
    actual_length: int | None


class Level(IntEnum):
    """RAID level identifiers."""

    LINEAR = -1
    # The rest is really just the RAID number
    RAID0 = 0
    RAID1 = 1
    RAID3 = 3
    RAID4 = 4
    RAID5 = 5
    RAID6 = 6
    RAID10 = 10


class Layout(IntEnum):
    """RAID layout identifiers.

    "Coincidentally" these align with Linux MD.
    """

    LEFT_ASYMMETRIC = 0  # Rotating Parity N with Data Restart
    RIGHT_ASYMMETRIC = 1  # Rotating Parity 0 with Data Restart
    LEFT_SYMMETRIC = 2  # Rotating Parity N with Data Continuation
    RIGHT_SYMMETRIC = 3  # Rotating Parity 0 with Data Continuation
    PARITY_0 = 4  # P or P,Q are initial devices
    PARITY_N = 5  # P or P,Q are final devices
    ROTATING_ZERO_RESTART = 8  # DDF PRL=6 RLQ=1
    ROTATING_N_RESTART = 9  # DDF PRL=6 RLQ=2
    ROTATING_N_CONTINUE = 10  # DDF PRL=6 RLQ=3
    LEFT_ASYMMETRIC_6 = 16
    RIGHT_ASYMMETRIC_6 = 17
    LEFT_SYMMETRIC_6 = 18
    RIGHT_SYMMETRIC_6 = 19
    PARITY_0_6 = 20
    PARITY_N_6 = PARITY_N


class LinearStream(MappingStream):
    """Implements a stream on a linear RAID set."""

    def __init__(self, virtual_disk: VirtualDisk):
        super().__init__()
        self.virtual_disk = virtual_disk
        self.disk_map: DiskMap = dict(sorted(virtual_disk.disk_map.items()))

        if len(self.disk_map) != virtual_disk.num_disks:
            raise RAIDError(f"Missing disks in linear RAID set {virtual_disk.uuid} ({virtual_disk.name})")

        offset = 0
        for disk_offset, disk in self.disk_map.values():
            self.add(offset, disk.size, disk.open(), disk_offset)
            offset += disk.size


class Zone(NamedTuple):
    zone_end: int
    dev_start: int
    devices: DiskMap


class RAID0Stream(AlignedStream):
    """Implements a stream on a RAID0 set."""

    def __init__(self, virtual_disk: VirtualDisk):
        self.virtual_disk = virtual_disk
        self.disk_map = dict(sorted(self.virtual_disk.disk_map.items()))

        if len(self.disk_map) != virtual_disk.num_disks:
            raise RAIDError(f"Missing disks in RAID0 set {virtual_disk.uuid} ({virtual_disk.name})")

        # Determine how many strip zones we need to construct
        # If a RAID0 set consists of devices with different sizes, additional strip zones
        # may exist on the larger devices but not on the smaller ones
        # Reference: create_strip_zones
        rounded_sizes: dict[PhysicalDisk, int] = {}
        stripe_size = virtual_disk.stripe_size
        num_strip_zones = 0
        for idx1, (_, dev1) in self.disk_map.items():
            rounded_sizes[dev1] = (dev1.size // stripe_size) * stripe_size

            has_same_size = False
            # Check if dev1 is unequal in size to the sizes of any of the previous devices
            # If so, this means an extra strip zone is present
            for idx2, (_, dev2) in self.disk_map.items():
                if idx1 == idx2:
                    break

                if rounded_sizes[dev1] == dev2.size:
                    has_same_size = True
                    break

            if not has_same_size:
                num_strip_zones += 1

        # Determine the smallest device
        smallest = None
        for _, dev in self.disk_map.values():
            if not smallest or rounded_sizes[dev] < rounded_sizes[smallest]:
                smallest = dev

        # Construct the strip zones
        zones = [Zone(rounded_sizes[smallest] * len(self.disk_map), 0, self.disk_map)]

        cur_zone_end = zones[0].zone_end
        for _ in range(1, num_strip_zones):
            zone_devices = {}
            dev_start = rounded_sizes[smallest]
            smallest = None

            # Look for the next smallest device, that is: the smallest device that is larger than the "dev_start" device
            for disk_idx, (data_offset, dev) in self.disk_map.items():
                if rounded_sizes[dev] <= dev_start:
                    continue

                zone_devices[disk_idx] = (data_offset, dev)
                if not smallest or rounded_sizes[dev] < rounded_sizes[smallest]:
                    smallest = dev

            num_dev = len(zone_devices)
            cur_size = (rounded_sizes[smallest] - dev_start) * num_dev

            cur_zone_end += cur_size

            zones.append(Zone(cur_zone_end, dev_start, zone_devices))

        self.zones = zones

        super().__init__(self.virtual_disk.size)

    def _find_zone(self, offset: int) -> tuple[Zone, int] | None:
        """Return the zone and the offset within that zone a given ``offset`` is in."""
        for i, zone in enumerate(self.zones):
            if offset < zone.zone_end:
                if i:
                    offset = offset - self.zones[i - 1].zone_end
                return zone, offset
        return None, None

    def _read(self, offset: int, length: int) -> bytes:
        result = []

        stripe_size = self.virtual_disk.stripe_size
        while length:
            zone, offset_in_zone = self._find_zone(offset)
            if zone is None:
                break

            read_offset = offset_in_zone
            if len(self.zones) == 1 or len(self.zones[1].devices) == 1:
                read_offset = offset

            stripe, offset_in_stripe = divmod(read_offset, stripe_size)
            chunk = offset_in_zone // (stripe_size * len(zone.devices))

            offset_in_device = (chunk * stripe_size) + offset_in_stripe
            device_start, device = zone.devices[stripe % len(zone.devices)]

            stripe_remaining = stripe_size - offset_in_stripe
            read_length = min(length, stripe_remaining)

            offset_on_disk = device.offset + device_start + offset_in_device
            device.fh.seek(offset_on_disk)
            result.append(device.fh.read(read_length))

            length -= read_length
            offset += read_length

        return b"".join(result)


class RAID456Stream(AlignedStream):
    """Implements a read-only stream on a RAID4/5/6 set, including degraded recovery.

    A RAID4/5 set can be opened with a single member missing: the missing data chunks are rebuilt
    from the P parity stripe (XOR of the surviving members). Every logical byte that is the result
    of such a rebuild is recorded in :attr:`reconstructed_ranges` and :meth:`reconstructed_bitmap`.

    A second failed member, or a member that returns a short read, is never padded with zero bytes.
    Instead a :class:`~dissect.volume.exceptions.RAIDStripeReadError` is raised carrying
    per-member :class:`MemberEvidence` for the failed interval.
    """

    def __init__(self, virtual_disk: VirtualDisk):
        self.virtual_disk = virtual_disk
        self.level = self.virtual_disk.level
        self.algorithm = self.virtual_disk.layout
        self.max_degraded = 2 if self.level == 6 else 1

        self.disk_map = self.virtual_disk.disk_map
        missing = self.virtual_disk.num_disks - len(self.disk_map)
        if missing > self.max_degraded:
            raise RAIDError(f"Missing disks in RAID{self.level} set {virtual_disk.uuid} ({virtual_disk.name})")

        # Logical (virtual volume) byte ranges that could only be produced through reconstruction.
        self.reconstructed_ranges: list[tuple[int, int]] = []

        super().__init__(self.virtual_disk.size, self.virtual_disk.stripe_size)

    def is_reconstructed(self, offset: int, length: int) -> bool:
        """Return whether every byte in the logical range ``[offset, offset + length)`` was rebuilt."""
        if length <= 0:
            return False
        end = offset + length
        covered = offset
        for range_start, range_end in self.reconstructed_ranges:
            if range_end <= covered or range_start > covered:
                if range_start >= end:
                    break
                continue
            covered = max(covered, min(range_end, end))
            if covered >= end:
                return True
        return False

    def reconstructed_bitmap(self, offset: int, length: int) -> bytearray:
        """Return a per-byte bitmap (1 = reconstructed, 0 = read directly) for a logical range."""
        end = offset + length
        bitmap = bytearray(length)
        for range_start, range_end in self.reconstructed_ranges:
            if range_end <= offset:
                continue
            if range_start >= end:
                break
            overlap_start = max(range_start, offset) - offset
            overlap_end = min(range_end, end) - offset
            bitmap[overlap_start:overlap_end] = b"\x01" * (overlap_end - overlap_start)
        return bitmap

    def _record_reconstructed(self, offset: int, length: int) -> None:
        """Record a logical byte range as reconstructed, merging it with adjacent ranges."""
        if length <= 0:
            return
        start, end = offset, offset + length
        if self.reconstructed_ranges and self.reconstructed_ranges[-1][1] == start:
            previous_start, _ = self.reconstructed_ranges[-1]
            self.reconstructed_ranges[-1] = (previous_start, end)
        else:
            self.reconstructed_ranges.append((start, end))

    def _read_member(self, disk_idx: int, member_offset: int, length: int, stripe_number: int = -1) -> bytes:
        """Read a chunk from a single member, turning any failure into forensic evidence.

        A missing member or a short read raises :class:`RAIDStripeReadError` instead of returning a
        zero-padded buffer, so a degraded read can never be mistaken for complete data.
        """
        entry = self.disk_map.get(disk_idx)
        if entry is None:
            evidence = [MemberEvidence(disk_idx, "member missing", member_offset, length, None)]
            raise RAIDStripeReadError(
                f"Required member disk {disk_idx} is missing from the RAID{self.level} set",
                evidence,
                member_offset,
                length,
                stripe_number,
            )

        dev_start, device = entry
        offset_on_disk = device.offset + dev_start + member_offset
        device.fh.seek(offset_on_disk)
        data = device.fh.read(length)
        if len(data) != length:
            evidence = [
                MemberEvidence(
                    disk_idx,
                    f"short read: expected {length} bytes, got {len(data)}",
                    offset_on_disk,
                    length,
                    len(data),
                )
            ]
            raise RAIDStripeReadError(
                f"Short read on member disk {disk_idx}: expected {length} bytes at {offset_on_disk:#x}, "
                f"got {len(data)}",
                evidence,
                member_offset,
                length,
                stripe_number,
            )
        return data

    def _rebuild_stripe(
        self,
        stripe: int,
        member_offset: int,
        chunk_length: int,
        logical_offset: int,
        logical_length: int,
    ) -> tuple[bytes, list[MemberEvidence]]:
        """Rebuild one missing chunk of a RAID4/5 stripe by XOR-ing every surviving member.

        RAID4/5 P parity is the XOR of *all* chunks in a stripe, so the missing chunk - data or
        parity - equals the XOR of every surviving member, the parity member included. Returns the
        rebuilt chunk together with any member evidence gathered while reading the survivors.
        Raises :class:`RAIDStripeReadError` if more than one member is unusable.
        """
        raid_disks = self.virtual_disk.num_disks
        rebuilt = 0
        evidence: list[MemberEvidence] = []
        surviving = 0

        for disk_idx in range(raid_disks):
            entry = self.disk_map.get(disk_idx)
            if entry is None:
                evidence.append(MemberEvidence(disk_idx, "member missing", member_offset, chunk_length, None))
                continue
            try:
                chunk = self._read_member(disk_idx, member_offset, chunk_length, stripe)
            except RAIDStripeReadError as err:
                evidence.extend(err.evidence)
                continue
            surviving += 1
            rebuilt ^= int.from_bytes(chunk, "big")

        required_survivors = raid_disks - 2
        if surviving < required_survivors or len(evidence) > 1:
            raise RAIDStripeReadError(
                f"Cannot rebuild stripe {stripe} of RAID{self.level} set: {len(evidence)} member(s) unusable, "
                f"a single failure is the maximum that P parity can recover",
                evidence,
                logical_offset,
                logical_length,
                stripe,
            )

        return rebuilt.to_bytes(chunk_length, "big"), evidence

    def _read(self, offset: int, length: int) -> bytes:
        result = []

        stripe_size = self.virtual_disk.stripe_size
        while length:
            stripe, offset_in_stripe, dd_idx, _, _, _ = self._get_stripe_read_info(offset)
            offset_in_device = stripe * stripe_size + offset_in_stripe

            stripe_remaining = stripe_size - offset_in_stripe
            read_length = min(length, stripe_remaining)

            if dd_idx in self.disk_map:
                # The requested data member is present, read it directly.
                data = self._read_member(dd_idx, offset_in_device, read_length, stripe)
            else:
                # The data member is missing. RAID4/5 can rebuild it from the P parity stripe.
                if self.level not in (Level.RAID4, Level.RAID5):
                    raise RAIDStripeReadError(
                        f"Degraded reconstruction of RAID{self.level} is not supported",
                        [MemberEvidence(dd_idx, "member missing", offset_in_device, read_length, None)],
                        offset,
                        read_length,
                        stripe,
                    )
                data, _ = self._rebuild_stripe(
                    stripe,
                    stripe * stripe_size,
                    stripe_size,
                    offset,
                    read_length,
                )
                data = data[offset_in_stripe : offset_in_stripe + read_length]
                self._record_reconstructed(offset, read_length)

            result.append(data)

            length -= read_length
            offset += read_length

        return b"".join(result)

    def _get_stripe_read_info(self, offset: int) -> tuple[int, int, int, int, int | None, bool]:
        """Calculate the stripe, offset in the stripe, data disk, parity disk and "Q" parity disk for a given sector."""
        # Reference: raid5_compute_sector
        stripe_size = self.virtual_disk.stripe_size
        raid_disks = self.virtual_disk.num_disks
        data_disks = raid_disks - self.max_degraded

        stripe_number, offset_in_stripe = divmod(offset, stripe_size)
        stripe, dd_idx = divmod(stripe_number, data_disks)

        pd_idx = None
        qd_idx = None
        ddf_layout = False

        if self.level == 4:
            pd_idx = data_disks

        elif self.level == 5:
            if self.algorithm == Layout.LEFT_ASYMMETRIC:
                pd_idx = data_disks - (stripe % raid_disks)
                if dd_idx >= pd_idx:
                    dd_idx += 1

            elif self.algorithm == Layout.RIGHT_ASYMMETRIC:
                pd_idx = stripe % raid_disks
                if dd_idx >= pd_idx:
                    dd_idx += 1

            elif self.algorithm == Layout.LEFT_SYMMETRIC:
                pd_idx = data_disks - (stripe % raid_disks)
                dd_idx = (pd_idx + 1 + dd_idx) % raid_disks

            elif self.algorithm == Layout.RIGHT_SYMMETRIC:
                pd_idx = stripe % raid_disks
                dd_idx = (pd_idx + 1 + dd_idx) % raid_disks

            elif self.algorithm == Layout.PARITY_0:
                pd_idx = 0
                dd_idx += 1

            elif self.algorithm == Layout.PARITY_N:
                pd_idx = data_disks

            else:
                raise RAIDError(f"Invalid RAID algorithm: {self.algorithm}")

        elif self.level == 6:
            if self.algorithm == Layout.LEFT_ASYMMETRIC:
                pd_idx = raid_disks - 1 - (stripe % raid_disks)
                qd_idx = pd_idx + 1
                if pd_idx == raid_disks - 1:
                    # Q D D D P
                    dd_idx += 1
                    qd_idx = 0
                elif dd_idx >= pd_idx:
                    # D D P Q D
                    dd_idx += 2

            elif self.algorithm == Layout.RIGHT_ASYMMETRIC:
                pd_idx = stripe % raid_disks
                qd_idx = pd_idx + 1
                if pd_idx == raid_disks - 1:
                    # Q D D D P
                    dd_idx += 1
                    qd_idx = 0
                elif dd_idx >= pd_idx:
                    # D D P Q D
                    dd_idx += 2

            elif self.algorithm == Layout.LEFT_SYMMETRIC:
                pd_idx = raid_disks - 1 - (stripe % raid_disks)
                qd_idx = (pd_idx + 1) % raid_disks
                dd_idx = (pd_idx + 2 + dd_idx) % raid_disks

            elif self.algorithm == Layout.RIGHT_SYMMETRIC:
                pd_idx = stripe % raid_disks
                qd_idx = (pd_idx + 1) % raid_disks
                dd_idx = (pd_idx + 2 + dd_idx) % raid_disks

            elif self.algorithm == Layout.PARITY_0:
                pd_idx = 0
                qd_idx = 1
                dd_idx += 2

            elif self.algorithm == Layout.PARITY_N:
                pd_idx = data_disks
                qd_idx = data_disks + 1

            elif self.algorithm == Layout.ROTATING_ZERO_RESTART:
                # Exactly the same as RIGHT_ASYMMETRIC, but or of blocks for computing Q is different
                pd_idx = stripe % raid_disks
                qd_idx = pd_idx + 1
                if pd_idx == raid_disks - 1:
                    dd_idx += 1
                    qd_idx = 0
                elif dd_idx >= pd_idx:
                    # D D P Q D
                    dd_idx += 2
                ddf_layout = True

            elif self.algorithm == Layout.ROTATING_N_RESTART:
                # Same as left_asymmetric, but first stripe is
                # D D D P Q  rather than
                # Q D D D P
                pd_idx = raid_disks - 1 - ((stripe + 1) % raid_disks)
                qd_idx = pd_idx + 1
                if pd_idx == raid_disks - 1:
                    # Q D D D P
                    dd_idx += 1
                    qd_idx = 0
                elif dd_idx >= pd_idx:
                    # D D P Q D
                    dd_idx += 2
                ddf_layout = True

            elif self.algorithm == Layout.ROTATING_N_CONTINUE:
                # Same as left_symmetric but Q is before P
                pd_idx = raid_disks - 1 - (stripe % raid_disks)
                qd_idx = (pd_idx + raid_disks - 1) % raid_disks
                dd_idx = (pd_idx + 1 + dd_idx) % raid_disks
                ddf_layout = True

            elif self.algorithm == Layout.LEFT_ASYMMETRIC_6:
                # RAID5 left_asymmetric, with Q on last device
                pd_idx = data_disks - (stripe % (raid_disks - 1))
                if dd_idx >= pd_idx:
                    dd_idx += 1
                qd_idx = raid_disks - 1

            elif self.algorithm == Layout.RIGHT_ASYMMETRIC_6:
                pd_idx = stripe % (raid_disks - 1)
                if dd_idx >= pd_idx:
                    dd_idx += 1
                    qd_idx = raid_disks - 1

            elif self.algorithm == Layout.LEFT_SYMMETRIC_6:
                pd_idx = data_disks - (stripe % (raid_disks - 1))
                dd_idx = (pd_idx + 1 + dd_idx) % (raid_disks - 1)
                qd_idx = raid_disks - 1

            elif self.algorithm == Layout.RIGHT_SYMMETRIC_6:
                pd_idx = stripe % (raid_disks - 1)
                dd_idx = (pd_idx + 1 + dd_idx) % (raid_disks - 1)
                qd_idx = raid_disks - 1

            elif self.algorithm == Layout.PARITY_0_6:
                pd_idx = 0
                dd_idx += 1
                qd_idx = raid_disks - 1
            else:
                raise RAIDError(f"Invalid RAID algorithm: {self.algorithm}")
        else:
            raise RAIDError(f"Invalid RAID level: {self.level}")

        return stripe, offset_in_stripe, dd_idx, pd_idx, qd_idx, ddf_layout


class RAID10Stream(AlignedStream):
    """Implements a stream on a RAID10 set."""

    def __init__(self, virtual_disk: VirtualDisk):
        self.virtual_disk = virtual_disk
        self.raid_disks = self.virtual_disk.num_disks
        self.disk_map = virtual_disk.disk_map

        # Reference: setup_geo
        layout = virtual_disk.layout
        self.near_copies = layout & 0xFF
        self.far_copies = (layout >> 8) & 0xFF
        self.far_offset = layout & (1 << 16)

        use_far_sets = layout >> 17
        if use_far_sets == 0:
            # Original layout
            self.far_set_size = self.raid_disks
        elif use_far_sets == 1:
            # "Improved" but bugged layout
            self.far_set_size = self.raid_disks // self.far_copies
        elif use_far_sets == 2:
            # "Improved" and fixed layout
            self.far_set_size = self.far_copies * self.near_copies
        else:
            raise ValueError("Invalid RAID10 layout: {layout:#x}")

        self.last_far_set_start = ((self.raid_disks / self.far_set_size) - 1) * self.far_set_size
        self.last_far_set_size = self.far_set_size + (self.raid_disks % self.far_set_size)

        super().__init__(self.virtual_disk.size, self.virtual_disk.stripe_size)

    def _read(self, offset: int, length: int) -> bytes:
        result = []

        stripe_size = self.virtual_disk.stripe_size
        while length:
            # Reference: __raid10_find_phys
            stripe_number, offset_in_stripe = divmod(offset, stripe_size)
            stripe, dev = divmod(stripe_number * self.near_copies, self.raid_disks)

            if self.far_offset:
                stripe *= self.far_copies
            device_start, device = self.disk_map[dev]

            stripe_remaining = stripe_size - offset_in_stripe
            read_length = min(length, stripe_remaining)

            offset_on_disk = device.offset + device_start + (stripe * stripe_size) + offset_in_stripe
            device.fh.seek(offset_on_disk)
            result.append(device.fh.read(length))

            length -= read_length
            offset += read_length

        return b"".join(result)
