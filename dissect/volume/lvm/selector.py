from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dissect.volume.exceptions import LVM2Error
from dissect.volume.lvm.physical import MetadataCopy

if TYPE_CHECKING:
    from dissect.volume.lvm.metadata import VolumeGroup
    from dissect.volume.lvm.physical import LVM2Device

log = logging.getLogger(__name__)


@dataclass
class SelectionRecord:
    """The decision made for a single :class:`~dissect.volume.lvm.physical.MetadataCopy`.

    Args:
        copy: The metadata copy this record describes.
        selected: Whether this copy was used as the authoritative Volume Group configuration.
        reason: Human readable explanation of why the copy was accepted or excluded.
    """

    copy: MetadataCopy
    selected: bool
    reason: str

    @property
    def device_id(self) -> str:
        return self.copy.device.id

    @property
    def metadata_area(self) -> int:
        return self.copy.metadata_area

    @property
    def copy_index(self) -> int:
        return self.copy.copy_index


@dataclass
class MetadataSelection:
    """Result of reconciling all metadata copies collected from the supplied Physical Volumes.

    Args:
        volume_group: The single, internally consistent Volume Group configuration that was selected.
        metadata: The raw parsed metadata dictionary backing :attr:`volume_group`.
        records: A decision record for every examined copy, valid or not, in a deterministic order.
    """

    volume_group: VolumeGroup
    metadata: dict
    records: list[SelectionRecord] = field(default_factory=list)

    @property
    def selected(self) -> SelectionRecord:
        """The record belonging to the copy that was actually used."""
        return next(record for record in self.records if record.selected)

    @property
    def excluded(self) -> list[SelectionRecord]:
        """All records that were examined but not used."""
        return [record for record in self.records if not record.selected]

    def report(self) -> str:
        """Render a human readable report listing the adopted and excluded copies and the reason for each."""
        lines = [
            f"Selected Volume Group {self.volume_group.name} (id={self.volume_group.id}, "
            f"seqno={self.volume_group.seqno}):"
        ]
        for record in self.records:
            prefix = "[USED]    " if record.selected else "[EXCLUDED]"
            lines.append(
                f"  {prefix} pv={record.device_id} mda={record.metadata_area} "
                f"copy={record.copy_index}: {record.reason}"
            )
        return "\n".join(lines)


def vg_identity(data: dict) -> tuple[str, str, int]:
    """Return the ``(vg_name, vg_id, seqno)`` identity from a parsed metadata dictionary.

    Raises:
        LVM2Error: If the dictionary does not contain a Volume Group section or the required ``id`` / ``seqno``
            fields.
    """
    vg_name = next((key for key, value in data.items() if isinstance(value, dict)), None)
    if vg_name is None:
        raise LVM2Error("Metadata does not contain a Volume Group section")

    vg = data[vg_name]
    vg_id = vg.get("id")
    seqno = vg.get("seqno")
    if vg_id is None or seqno is None:
        raise LVM2Error("Volume Group metadata is missing its id or seqno")

    return vg_name, vg_id, seqno


def stable_sort_key(copy: MetadataCopy) -> tuple[str, int, int, int]:
    """Deterministic ordering based on physical volume identity rather than input order."""
    return (copy.device.id, copy.metadata_area, copy.copy_index, copy.offset)


def _choose_copy(copies: list[MetadataCopy]) -> tuple[MetadataCopy, str]:
    """Pick one copy among the copies sharing the highest sequence number.

    Copies carrying byte-for-byte identical text form a "witness" group. The configuration present on the most
    distinct physical volumes is preferred because a state committed to multiple PVs is the most likely truth.
    Any remaining tie is resolved with :func:`stable_sort_key`, which is independent of input ordering.
    """
    ordered = sorted(copies, key=stable_sort_key)

    witnesses: dict[str, set[str]] = {}
    by_text: dict[str, list[MetadataCopy]] = {}
    for copy in ordered:
        witnesses.setdefault(copy.text, set()).add(copy.device.id)
        by_text.setdefault(copy.text, []).append(copy)

    # Highest number of distinct PV witnesses first, then the deterministic stable key.
    best_text = min(
        witnesses,
        key=lambda text: (-len(witnesses[text]), stable_sort_key(by_text[text][0])),
    )
    chosen = by_text[best_text][0]
    witness_count = len(witnesses[best_text])

    if len(witnesses) == 1:
        reason = f"only configuration present at this seqno, identical on {witness_count} physical volume(s)"
    else:
        reason = (
            f"chosen from {len(witnesses)} conflicting configurations sharing this seqno; corroborated by "
            f"{witness_count} physical volume(s), deterministic tie break applied"
        )
    return chosen, reason


def select_metadata(devices: list[LVM2Device]) -> MetadataSelection:
    """Select a single, internally consistent Volume Group configuration from all available metadata copies.

    Every metadata copy is treated as an indivisible whole: logical and physical volumes are never combined
    across copies (and therefore across generations), so a non-existent Logical Volume can never be assembled
    from mixed metadata. The decision is based solely on Volume Group identity and sequence number and is
    independent of the order in which devices are supplied.

    Selection strategy:

    * Copies that fail their checksum, cannot be parsed, are marked ignored or lack a Volume Group section are
      excluded with an explicit reason.
    * Valid copies are grouped by Volume Group UUID; only a single Volume Group is supported.
    * Within a Volume Group the highest ``seqno`` (newest generation) wins; older generations are excluded.
    * Among equal-``seqno`` copies, identical text counts as corroborating witnesses and conflicting text is
      resolved deterministically (see :func:`_choose_copy`).

    Args:
        devices: The Physical Volumes to inspect.

    Returns:
        A :class:`MetadataSelection` holding the chosen configuration and a full decision trail.

    Raises:
        LVM2Error: If no usable copy exists, multiple distinct Volume Groups are found, or a copy lacks the
            fields required for identification.
    """
    from dissect.volume.lvm.metadata import VolumeGroup

    records: list[SelectionRecord] = []
    valid_copies: list[MetadataCopy] = []

    # Sorting devices up front, together with stable_sort_key, makes every later ordering deterministic.
    for device in sorted(devices, key=lambda dev: dev.id):
        for copy in device.metadata_copies:
            if copy.ignored:
                records.append(SelectionRecord(copy, False, copy.error or "raw location is marked as ignored"))
            elif not copy.valid:
                records.append(SelectionRecord(copy, False, copy.error or "invalid metadata copy"))
            else:
                valid_copies.append(copy)

    groups: dict[str, list[MetadataCopy]] = {}
    for copy in sorted(valid_copies, key=stable_sort_key):
        try:
            _, vg_id, _ = vg_identity(copy.data)
        except LVM2Error as err:
            records.append(SelectionRecord(copy, False, str(err)))
            continue
        groups.setdefault(vg_id, []).append(copy)

    if not groups:
        total = sum(len(device.metadata_copies) for device in devices)
        invalid = total - len(valid_copies)
        raise LVM2Error(
            f"No valid LVM2 metadata could be loaded from {len(devices)} physical volume(s); "
            f"{invalid} of {total} metadata cop(ies) failed validation"
        )

    if len(groups) > 1:
        identities = sorted(groups)
        raise LVM2Error(f"Found multiple distinct volume groups, expected only one: {identities}")

    vg_copies = sorted(next(iter(groups.values())), key=stable_sort_key)
    highest_seqno = max(vg_identity(copy.data)[2] for copy in vg_copies)
    newest = [copy for copy in vg_copies if vg_identity(copy.data)[2] == highest_seqno]
    older = [copy for copy in vg_copies if vg_identity(copy.data)[2] != highest_seqno]

    for copy in older:
        records.append(
            SelectionRecord(
                copy, False, f"older generation seqno={vg_identity(copy.data)[2]} < selected seqno={highest_seqno}"
            )
        )

    selected_copy, selection_reason = _choose_copy(newest)

    for copy in newest:
        if copy is selected_copy:
            continue
        if copy.text == selected_copy.text:
            records.append(
                SelectionRecord(
                    copy, False, f"identical copy of selected generation seqno={highest_seqno}; corroborating witness"
                )
            )
        else:
            records.append(
                SelectionRecord(copy, False, f"same seqno={highest_seqno} but different content; not selected")
            )

    selected_record = SelectionRecord(selected_copy, True, selection_reason)
    records = [selected_record, *sorted(records, key=lambda record: stable_sort_key(record.copy))]

    vg_name, _, _ = vg_identity(selected_copy.data)
    volume_group = VolumeGroup.from_dict(selected_copy.data[vg_name], name=vg_name)

    selection = MetadataSelection(volume_group, selected_copy.data, records)
    log.debug("\n%s", selection.report())
    return selection
