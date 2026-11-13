from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypedDict

from dissect.volume.vinum.c_vinum import c_vinum

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from datetime import datetime

log = logging.getLogger(__name__)
log.setLevel(os.getenv("DISSECT_LOG_VINUM", "CRITICAL"))


@dataclass
class ConfigSource:
    """Location of a configuration object in the original, unmodified config block."""

    line: int
    raw: bytes
    offset: int = 0
    disk: bytes | None = None


@dataclass
class Volume:
    """The representation of a Vinum Volume.

    A Vinum Volume defines a single RAID set. One or more Vinum Plexes can be
    part of a Volume.
    """

    timestamp: datetime
    name: bytes
    state: VolumeState | None = None
    source: ConfigSource | None = None


@dataclass
class Plex:
    """The representation of a Vinum Plex.

    A Vinum Plex can be thought of as one of the individual disks in a mirrored
    array. One or more Vinum SDs can be part of a Plex. The Plex defines the
    type of RAID in which these SDs are organized.
    """

    timestamp: datetime
    name: bytes | None = None
    org: PlexOrg | None = None
    stripesize: int | None = None
    volume: bytes | None = None
    state: PlexState | None = None
    source: ConfigSource | None = None


@dataclass
class SD:
    """The representation of a Vinum SD.

    A Vinum SD contains information about the actual physical disk and points
    to the device of this disk.
    """

    timestamp: datetime
    drive: bytes
    name: bytes | None = None
    # length is the size in bytes of the data section on disk, so without any
    # vinum headers etc.
    length: int | None = None
    # driveoffset is the start of the data section on disk in bytes.
    driveoffset: int | None = None
    plex: bytes | None = None
    # plexoffset is the offset of the data section of this disk within the plex in
    # bytes, e.g.: the first disk always starts at offset 0, if the size of its
    # data section (SD.length) is 1024b then the plexoffset for the second disk
    # will be 1024.
    plexoffset: int | None = None
    state: SDState | None = None
    source: ConfigSource | None = None


class IssueSeverity(str, Enum):
    """Severity of a Vinum configuration diagnostic."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class VinumIssue:
    """A read-only diagnostic for a Vinum configuration object.

    ``source`` points at the exact configuration line the issue was derived from.
    ``volume``, ``plex``, ``sd`` and ``drive`` identify the objects involved.
    """

    severity: IssueSeverity
    code: str
    message: str
    source: ConfigSource | None = None
    sources: tuple[ConfigSource, ...] = ()
    volume: bytes | None = None
    plex: bytes | None = None
    sd: bytes | None = None
    drive: bytes | None = None

    @property
    def line(self) -> int | None:
        return self.source.line if self.source else None

    @property
    def lines(self) -> tuple[int, ...]:
        """All configuration lines this issue refers to, primary line first."""
        lines = [source.line for source in self.sources]
        if self.source is not None:
            lines.insert(0, self.source.line)
        return tuple(dict.fromkeys(lines))

    def __str__(self) -> str:
        location = "unknown location"
        if self.source is not None:
            location = f"line {self.source.line}"
            if self.source.disk is not None:
                location = f"{location} on drive {_display(self.source.disk)}"

        objects = []
        if self.volume is not None:
            objects.append(f"volume {_display(self.volume)}")
        if self.plex is not None:
            objects.append(f"plex {_display(self.plex)}")
        if self.sd is not None:
            objects.append(f"sd {_display(self.sd)}")
        if self.drive is not None:
            objects.append(f"drive {_display(self.drive)}")
        subject = f" ({', '.join(objects)})" if objects else ""

        return f"{self.severity.value}: {location}: {self.code}{subject}: {self.message}"


class VolumeAvailability(str, Enum):
    """Whether a volume can be opened or is only retained as configuration evidence."""

    READABLE = "readable"
    EVIDENCE_ONLY = "evidence-only"


@dataclass
class VolumeAssessment:
    """Result of checking one Volume, its Plexes and their SDs."""

    volume: Volume
    status: VolumeAvailability = VolumeAvailability.EVIDENCE_ONLY
    reason: str | None = None
    issues: list[VinumIssue] = field(default_factory=list)
    plexes: list[Plex] = field(default_factory=list)
    sds: dict[bytes, list[SD]] = field(default_factory=dict)

    @property
    def name(self) -> bytes:
        return self.volume.name

    @property
    def readable(self) -> bool:
        return self.status is VolumeAvailability.READABLE


class MergedVinumConfigs(TypedDict):
    """Configuration objects merged by name across all configuration blocks."""

    volumes: dict[bytes, Volume]
    plexes: dict[bytes, Plex]
    sds: dict[bytes, SD]


@dataclass
class VinumDiagnostics:
    """Diagnostics for a merged Vinum configuration.

    ``volumes`` contains an assessment for every Volume found in the configuration,
    including volumes that can only be preserved as configuration evidence.
    ``merged`` holds the objects the analyzer actually evaluated.
    """

    volumes: list[VolumeAssessment] = field(default_factory=list)
    issues: list[VinumIssue] = field(default_factory=list)
    merged: MergedVinumConfigs = field(
        default_factory=lambda: {"volumes": {}, "plexes": {}, "sds": {}}
    )

    @property
    def readable_volumes(self) -> list[bytes]:
        return [assessment.name for assessment in self.volumes if assessment.readable]

    @property
    def evidence_only_volumes(self) -> list[bytes]:
        return [assessment.name for assessment in self.volumes if not assessment.readable]

    @property
    def readable(self) -> dict[bytes, VolumeAssessment]:
        return {assessment.name: assessment for assessment in self.volumes if assessment.readable}

    @property
    def evidence_only(self) -> dict[bytes, VolumeAssessment]:
        return {assessment.name: assessment for assessment in self.volumes if not assessment.readable}

    def assessment(self, name: bytes) -> VolumeAssessment | None:
        for assessment in self.volumes:
            if assessment.name == name:
                return assessment
        return None

    def format_report(self) -> str:
        """Return a human readable readable/evidence-only summary."""
        readable = [assessment for assessment in self.volumes if assessment.readable]
        evidence = [assessment for assessment in self.volumes if not assessment.readable]

        lines = ["Readable volumes:"]
        lines.extend(f"  - {_display(assessment.name)}" for assessment in readable)
        if not readable:
            lines.append("  (none)")

        lines.append("Evidence-only volumes:")
        if not evidence:
            lines.append("  (none)")
        for assessment in evidence:
            lines.append(f"  - {_display(assessment.name)}: {assessment.reason or 'unavailable'}")
            lines.extend(f"      {issue}" for issue in assessment.issues)

        return "\n".join(lines)


class ParseError(Exception):
    pass


class BytesDefaultEnum(bytes, Enum):
    @classmethod
    def _missing_(cls, value: bytes) -> Any:
        return cls._default


class VolumeState(BytesDefaultEnum):
    DOWN = auto()
    UP = b"up"

    _default = DOWN


class PlexState(BytesDefaultEnum):
    DOWN = auto()
    UP = b"up"
    INITIALIZING = b"initializing"
    DEGRADED = b"degraded"
    GROWABLE = b"growable"

    _default = DOWN


class PlexOrg(BytesDefaultEnum):
    DISORG = auto()
    CONCAT = b"concat"
    STRIPED = b"striped"
    RAID5 = b"raid5"

    _default = DISORG


class SDState(BytesDefaultEnum):
    DOWN = auto()
    UP = b"up"
    INITIALIZING = b"initializing"
    DEGRADED = b"degraded"
    GROWABLE = b"growable"

    _default = DOWN


def _parse_size(size: bytes) -> int:
    # Only the first byte after the numerals (and optional minus sign) should
    # be considered.
    postfix = size.lstrip(b"-0123456789")
    numeral = size[: -len(postfix)] if postfix else size
    unit = postfix[:1]

    try:
        size = int(numeral)
    except ValueError:
        # If there are no numerals (numeral is empty or the minus sign), the
        # size should be parsed as 0.
        size = 0
    else:
        if unit:
            # Invalid unites should be ignored and size is returned as is.
            if unit in (b"b", b"B", b"s", b"S"):
                size = size * 512  # Yes also for b/B
            elif unit in (b"k", b"K"):
                size = size * 1024
            elif unit in (b"m", b"M"):
                size = size * 1024 * 1024
            elif unit in (b"g", b"G"):
                size = size * 1024 * 1024 * 1024

    return size


def _display(value: bytes | str) -> str:
    if isinstance(value, str):
        return value
    return value.decode(errors="backslashreplace")


def _parse_volume_config(
    config_time: datetime, tokens: Iterable[bytes], source: ConfigSource | None = None
) -> Volume | None:
    volume = None
    name = None
    state = None

    tokens = iter(tokens)
    token = next(tokens, None)
    try:
        while token is not None:
            if token == b"state":
                state = VolumeState(next(tokens))
            else:
                name = token
            token = next(tokens, None)
    except StopIteration:
        log.debug("No value for token %r, ignoring volume config", token)
    else:
        if name is None:
            log.debug("No name found for volume, ignoring volume config")
        else:
            volume = Volume(
                timestamp=config_time,
                name=name,
                state=state,
                source=source,
            )

    return volume


def _parse_plex_config(
    config_time: datetime, tokens: Iterable[bytes], source: ConfigSource | None = None
) -> Plex | None:
    plex = None
    name = None
    org = None
    stripesize = None
    volume = None
    state = None

    tokens = iter(tokens)
    token = next(tokens, None)
    try:
        while token is not None:
            if token == b"name":
                name = next(tokens)
            elif token == b"org":
                org = PlexOrg(next(tokens))
                if org == PlexOrg.RAID5 or org == PlexOrg.STRIPED:
                    stripesize = _parse_size(next(tokens))
                    # the kernel parser only checks on == 0, but < 0 also seems unreasonable
                    if stripesize <= 0:
                        raise ParseError(f"Invalid stripesize: {stripesize}")  # noqa: TRY301
            elif token == b"vol" or token == b"volume":
                volume = next(tokens)
            elif token == b"state":
                state = PlexState(next(tokens))
            else:
                raise ParseError(f"Unknown token {token}")  # noqa: TRY301

            token = next(tokens, None)

    except (StopIteration, ParseError) as err:
        if isinstance(err, StopIteration):
            log.debug("No value for token %r, ignoring plex config", token)
        else:
            log.debug("%s, ignoring plex config", err)

    else:
        plex = Plex(
            timestamp=config_time,
            name=name,
            org=org,
            stripesize=stripesize,
            volume=volume,
            state=state,
            source=source,
        )

    return plex


def _parse_sd_config(
    config_time: datetime, tokens: Iterable[bytes], source: ConfigSource | None = None
) -> SD | None:
    sd = None
    name = None
    drive = None
    length = None
    driveoffset = None
    plex = None
    plexoffset = None
    state = None

    tokens = iter(tokens)
    token = next(tokens, None)
    try:
        while token is not None:
            if token == b"name":
                name = next(tokens)
            elif token == b"drive":
                drive = next(tokens)
            elif token == b"len" or token == b"length":
                length = _parse_size(next(tokens))
                if length < 0:
                    length = -1
            elif token == b"driveoffset":
                driveoffset = _parse_size(next(tokens))
                if driveoffset != 0 and driveoffset < c_vinum.GV_DATA_START:
                    raise ParseError(f"Invalid driveoffset: {driveoffset}")  # noqa: TRY301
            elif token == b"plex":
                plex = next(tokens)
            elif token == b"plexoffset":
                plexoffset = _parse_size(next(tokens))
                if plexoffset < 0:
                    raise ParseError(f"Invalid plexoffset: {plexoffset}")  # noqa: TRY301
            elif token == b"state":
                state = SDState(next(tokens))
            else:
                raise ParseError(f"Unknown token {token}")  # noqa: TRY301

            token = next(tokens, None)

    except (StopIteration, ParseError) as err:
        if isinstance(err, StopIteration):
            log.debug("No value for token %r, ignoring sd config", token)
        else:
            log.debug("%s, ignoring sd config", err)

    else:
        if drive is None:
            log.debug("No drive found for sd, ignoring sd config")
        else:
            sd = SD(
                timestamp=config_time,
                name=name,
                drive=drive,
                length=length,
                driveoffset=driveoffset,
                plex=plex,
                plexoffset=plexoffset,
                state=state,
                source=source,
            )

    return sd


def get_char(line: bytes, idx: int) -> bytes:
    """Return a single byte bytestring at index ``idx`` in ``line``.

    If the index is outside of the bounaries of ``line``, an empty bytestring
    will be returned.
    """
    char = b""
    if idx >= 0 and idx < len(line):
        char = line[idx : idx + 1]  # this makes sure we get a single byte bytestring
    return char


class TokenizeError(Exception):
    pass


def tokenize(line: bytes) -> iter[bytes]:
    """Yield individual tokens from a vinum config line.

    This token parser is constructed to be equivalent to the token parser used in the
    FreeBSD kernel code. There are a few caveats though:

    - it expects lines to be pre-splitted on newline and null-byte characters
    - it does not attempt to parse quoted tokens, as the code in the kernel parser is
      buggy and will always lead to an error condition (it will mimick the error condition
      though).
    """
    whitespace = {b" ", b"\t"}
    quotes = {b'"', b"'"}
    comment = {b"#"}
    eol = {b""}
    end_of_list = eol.union(comment)
    end_of_token = whitespace.union(eol)

    token = b""
    idx = 0
    while True:
        char = get_char(line, idx)

        while char in whitespace:
            # Remove leading whitespace up to the next token or end_of_list condition
            idx += 1
            char = get_char(line, idx)

        if char in end_of_list:
            # We are at the end of the token list (a comment or end of line).
            break

        if char in quotes:
            # Encountering a quoted token will always lead to an error
            # condition in the (Free)BSD vinum kernel code. This is a bug in
            # that code, which we mimick here.
            raise TokenizeError(f"Found quoted token at index {idx}")

        while char not in end_of_token:
            # Add characters to the token until we encounter a stop condition.
            # Note that comment and quote characters are allowed in a token as
            # long as they are not preceded by whitespace.
            token += char
            idx += 1
            char = get_char(line, idx)

        if token:
            yield token
            token = b""

        idx += 1


class VinumConfigs(TypedDict):
    """Vinum configurations."""

    volumes: list[Volume]
    plexes: list[Plex]
    sds: list[SD]


@dataclass
class ParsedVinumConfig:
    """A parsed configuration block together with any line level parse issues."""

    volumes: list[Volume] = field(default_factory=list)
    plexes: list[Plex] = field(default_factory=list)
    sds: list[SD] = field(default_factory=list)
    issues: list[VinumIssue] = field(default_factory=list)

    def as_dict(self) -> VinumConfigs:
        return {"volumes": self.volumes, "plexes": self.plexes, "sds": self.sds}


RE_CONFIG_EOL = re.compile(b"[\x00\n]")


TOKEN_CONFIG_MAP = {
    b"volume": "volumes",
    b"plex": "plexes",
    b"sd": "sds",
}


def _iter_config_lines(config: bytes) -> Iterable[tuple[int, int, bytes]]:
    """Yield ``(line_number, byte_offset, raw_line)`` for a config block."""
    start = 0
    line_number = 1
    for match in RE_CONFIG_EOL.finditer(config):
        yield line_number, start, config[start : match.start()]
        start = match.end()
        line_number += 1
    yield line_number, start, config[start:]


def _parse_vinum_config(config_time: datetime, config: bytes, disk: bytes | None = None) -> ParsedVinumConfig:
    parsed = ParsedVinumConfig()

    for line_number, offset, line in _iter_config_lines(config):
        source = ConfigSource(line=line_number, raw=line, offset=offset, disk=disk)

        try:
            tokens = tokenize(line)
            token = next(tokens, None)
            if token is None:
                # We encountered a line without tokens (empty, just whitespace or # comments)
                continue
            if token == b"volume":
                parsed_config = _parse_volume_config(config_time, tokens, source)
            elif token == b"plex":
                parsed_config = _parse_plex_config(config_time, tokens, source)
            elif token == b"sd":
                parsed_config = _parse_sd_config(config_time, tokens, source)
            else:
                parsed_config = None
                log.debug("Unknown config type in line: %r, ignoring config line", line)

            if parsed_config:
                config_type = TOKEN_CONFIG_MAP[token]
                getattr(parsed, config_type).append(parsed_config)
            else:
                log.debug("Invalid config line %r", line)
                log.debug("Ignoring this line and the rest of the config data")
                parsed.issues.append(
                    VinumIssue(
                        severity=IssueSeverity.ERROR,
                        code="invalid_config_line",
                        message=f"unable to parse {_display(token)} configuration line",
                        source=source,
                    )
                )
                parsed.issues.append(
                    VinumIssue(
                        severity=IssueSeverity.WARNING,
                        code="config_truncated",
                        message="the remainder of this configuration block was ignored",
                        source=source,
                    )
                )
                break
        except TokenizeError as err:
            log.debug("Invalid config line %r: %s", line, err)
            log.debug("Ignoring this line and the rest of the config data")
            parsed.issues.append(
                VinumIssue(
                    severity=IssueSeverity.ERROR,
                    code="tokenize_error",
                    message=str(err),
                    source=source,
                )
            )
            parsed.issues.append(
                VinumIssue(
                    severity=IssueSeverity.WARNING,
                    code="config_truncated",
                    message="the remainder of this configuration block was ignored",
                    source=source,
                )
            )
            break

    return parsed


def parse_vinum_config(config_time: datetime, config: bytes) -> VinumConfigs:
    """Parse the on-disk vinum configuration.

    Parsing forgiveness and strictness is implemented in the same way as in the vinum kernel code:

    Lines with an unknown configuration ``type`` (not ``volume``, ``plex`` or ``sd``), are ignored.

    Lines that fail to parse due to:
      - no name present
      - no value present for a token
      - unknown token name
      - a tokenization error

    will fail that line and the subsequent lines (rest of the config) to not being parsed.
    """
    return _parse_vinum_config(config_time, config).as_dict()


def parse_vinum_config_with_issues(
    config_time: datetime, config: bytes, disk: bytes | None = None
) -> ParsedVinumConfig:
    """Parse a configuration block while keeping line level diagnostics.

    This is the diagnostic counterpart of :func:`parse_vinum_config`. The parsed
    objects are the same, but each one keeps a :class:`ConfigSource` pointing at
    the original configuration line, and parse failures are reported as
    :class:`VinumIssue` objects instead of only being logged.
    """
    return _parse_vinum_config(config_time, config, disk=disk)


@dataclass
class _Occurrence:
    item: Volume | Plex | SD
    index: int
    disk: bytes | None


def _singular(kind: str) -> str:
    return {"volumes": "volume", "plexes": "plex", "sds": "sd"}[kind]


def _item_key(kind: str, item: Volume | Plex | SD) -> tuple[Any, ...]:
    if kind == "volumes":
        return (item.name, item.state)
    if kind == "plexes":
        return (item.name, item.org, item.stripesize, item.volume, item.state)
    return (
        item.name,
        item.drive,
        item.length,
        item.driveoffset,
        item.plex,
        item.plexoffset,
        item.state,
    )


def _item_objects(kind: str, item: Volume | Plex | SD) -> dict[str, bytes | None]:
    if kind == "volumes":
        return {"volume": item.name, "plex": None, "sd": None}
    if kind == "plexes":
        return {"volume": item.volume, "plex": item.name, "sd": None}
    return {"volume": None, "plex": item.plex, "sd": item.name}


def _merge_items(
    kind: str, items: list[Volume | Plex | SD], issues: list[VinumIssue]
) -> tuple[dict[bytes | None, Volume | Plex | SD], set[bytes | None]]:
    """Merge objects by name, keeping the newest definition and flagging ambiguity."""
    grouped: dict[bytes | None, list[_Occurrence]] = defaultdict(list)
    for index, item in enumerate(items):
        grouped[item.name].append(
            _Occurrence(item=item, index=index, disk=item.source.disk if item.source else None)
        )

    merged: dict[bytes | None, Volume | Plex | SD] = {}
    conflicted: set[bytes | None] = set()

    for name, occurrences in grouped.items():
        if name is None:
            # Nameless objects can not be referenced reliably.
            merged[None] = occurrences[-1].item
            continue

        # The kernel only updates a configuration if it is newer. Within one
        # block, repeated definitions are an ambiguity the operator must see.
        by_disk: dict[bytes | None, list[_Occurrence]] = defaultdict(list)
        for occurrence in occurrences:
            by_disk[occurrence.disk].append(occurrence)

        for disk_occurrences in by_disk.values():
            if len(disk_occurrences) < 2:
                continue
            disk_sources = tuple(
                occurrence.item.source for occurrence in disk_occurrences if occurrence.item.source is not None
            )
            if len({_item_key(kind, occurrence.item) for occurrence in disk_occurrences}) > 1:
                conflicted.add(name)
                issues.append(
                    VinumIssue(
                        severity=IssueSeverity.ERROR,
                        code=f"duplicate_{_singular(kind)}_conflict",
                        message=f"conflicting definitions for {_singular(kind)} {_display(name)}",
                        source=disk_sources[0] if disk_sources else None,
                        sources=disk_sources,
                        **_item_objects(kind, occurrences[0].item),
                    )
                )
            else:
                issues.append(
                    VinumIssue(
                        severity=IssueSeverity.INFO,
                        code=f"duplicate_{_singular(kind)}",
                        message=f"repeated identical definition for {_singular(kind)} {_display(name)}",
                        source=disk_sources[0] if disk_sources else None,
                        sources=disk_sources,
                        **_item_objects(kind, occurrences[0].item),
                    )
                )

        newest_timestamp = max(occurrence.item.timestamp for occurrence in occurrences)
        newest = [occurrence for occurrence in occurrences if occurrence.item.timestamp == newest_timestamp]
        if len({_item_key(kind, occurrence.item) for occurrence in newest}) > 1 and name not in conflicted:
            conflicted.add(name)
            newest_sources = tuple(
                occurrence.item.source for occurrence in newest if occurrence.item.source is not None
            )
            issues.append(
                VinumIssue(
                    severity=IssueSeverity.ERROR,
                    code=f"duplicate_{_singular(kind)}_conflict",
                    message=f"equally recent conflicting definitions for {_singular(kind)} {_display(name)}",
                    source=newest_sources[0] if newest_sources else None,
                    sources=newest_sources,
                    **_item_objects(kind, newest[0].item),
                )
            )

        chosen = max(occurrences, key=lambda occurrence: (occurrence.item.timestamp, -occurrence.index))
        merged[name] = chosen.item

    return merged, conflicted


def _implied_volume(plex_name: bytes | None) -> bytes | None:
    if plex_name is None:
        return None
    match = re.match(rb"^(?P<volume>.+)\.p\d+$", plex_name)
    return match.group("volume") if match else None


def _implied_plex(sd_name: bytes | None) -> bytes | None:
    if sd_name is None:
        return None
    match = re.match(rb"^(?P<plex>.+\.p\d+)\.s\d+$", sd_name)
    return match.group("plex") if match else None


def _plex_size(plex: Plex, sds: list[SD]) -> int | None:
    if not sds or any(sd.length is None for sd in sds):
        return None
    if plex.org == PlexOrg.RAID5:
        return (len(sds) - 1) * sds[0].length
    if plex.org == PlexOrg.STRIPED:
        return len(sds) * sds[0].length
    return sum(sd.length for sd in sds)


def _plex_problem_issues(
    plex: Plex,
    volume: bytes | None,
    implied_volume: bytes | None,
    volumes: Mapping[bytes, Volume],
    conflicted: bool,
) -> tuple[list[VinumIssue], bool]:
    issues: list[VinumIssue] = []
    usable = True

    def add(severity: IssueSeverity, code: str, message: str) -> None:
        nonlocal usable
        issues.append(
            VinumIssue(
                severity=severity,
                code=code,
                message=message,
                source=plex.source,
                volume=volume,
                plex=plex.name,
            )
        )
        if severity is IssueSeverity.ERROR:
            usable = False

    if plex.name is None:
        add(IssueSeverity.ERROR, "plex_missing_name", "plex has no name and can not be referenced")
        return issues, False

    if conflicted:
        add(IssueSeverity.ERROR, "duplicate_plex_conflict", "plex name has conflicting definitions")

    if volume is None:
        add(IssueSeverity.ERROR, "plex_unknown_volume", "plex has no volume association")
    elif volume not in volumes:
        add(IssueSeverity.ERROR, "plex_unknown_volume", f"plex references unknown volume {_display(volume)}")

    if implied_volume is not None and volume is not None and implied_volume != volume and implied_volume in volumes:
        add(
            IssueSeverity.ERROR,
            "plex_volume_ownership_mismatch",
            f"plex name implies volume {_display(implied_volume)} but it claims {_display(volume)}",
        )

    if plex.org is None or plex.org == PlexOrg.DISORG:
        add(IssueSeverity.ERROR, "plex_unknown_org", "plex has no supported organisation")
    elif plex.org in (PlexOrg.STRIPED, PlexOrg.RAID5) and (plex.stripesize is None or plex.stripesize <= 0):
        add(IssueSeverity.ERROR, "plex_invalid_stripesize", "striped plex has an invalid stripesize")

    if plex.state is None:
        add(IssueSeverity.WARNING, "plex_state_unknown", "plex has no state, assuming it is usable")
    elif plex.state is PlexState.DOWN:
        add(IssueSeverity.ERROR, "plex_down", "plex is down")
    elif plex.state is PlexState.INITIALIZING:
        add(IssueSeverity.ERROR, "plex_initializing", "plex is still initializing")
    elif plex.state is PlexState.DEGRADED:
        add(IssueSeverity.WARNING, "plex_degraded", "plex is degraded")

    return issues, usable


def _sd_problem_issues(
    sd: SD,
    plex_name: bytes | None,
    implied_plex: bytes | None,
    volume: bytes | None,
    plexes: Mapping[bytes, Plex],
    drives: set[bytes] | None,
    conflicted: bool,
) -> tuple[list[VinumIssue], bool]:
    issues: list[VinumIssue] = []
    usable = True

    def add(severity: IssueSeverity, code: str, message: str) -> None:
        nonlocal usable
        issues.append(
            VinumIssue(
                severity=severity,
                code=code,
                message=message,
                source=sd.source,
                volume=volume,
                plex=plex_name,
                sd=sd.name,
                drive=sd.drive,
            )
        )
        if severity is IssueSeverity.ERROR:
            usable = False

    if sd.name is None:
        add(IssueSeverity.ERROR, "sd_missing_name", "sd has no name and can not be referenced")
        return issues, False

    if conflicted:
        add(IssueSeverity.ERROR, "duplicate_sd_conflict", "sd name has conflicting definitions")

    if plex_name is None:
        add(IssueSeverity.ERROR, "sd_unknown_plex", "sd has no plex association")
    elif plex_name not in plexes:
        add(IssueSeverity.ERROR, "sd_unknown_plex", f"sd references unknown plex {_display(plex_name)}")

    if implied_plex is not None and plex_name is not None and implied_plex != plex_name and implied_plex in plexes:
        add(
            IssueSeverity.ERROR,
            "sd_plex_ownership_mismatch",
            f"sd name implies plex {_display(implied_plex)} but it claims {_display(plex_name)}",
        )

    if drives is not None and sd.drive not in drives:
        add(IssueSeverity.ERROR, "sd_missing_drive", f"physical drive {_display(sd.drive)} is not available")

    if sd.length is None:
        add(IssueSeverity.ERROR, "sd_missing_length", "sd has no length")
    elif sd.length <= 0:
        add(IssueSeverity.ERROR, "sd_invalid_length", f"sd has an invalid length: {sd.length}")

    if sd.plexoffset is None:
        add(IssueSeverity.ERROR, "sd_missing_plexoffset", "sd has no plexoffset")
    elif sd.plexoffset < 0:
        add(IssueSeverity.ERROR, "sd_invalid_plexoffset", f"sd has an invalid plexoffset: {sd.plexoffset}")

    if sd.driveoffset is None:
        add(
            IssueSeverity.WARNING,
            "sd_driveoffset_unknown",
            "sd has no driveoffset, assuming the data area starts at 0",
        )

    if sd.state is None:
        add(IssueSeverity.WARNING, "sd_state_unknown", "sd has no state, assuming it is usable")
    elif sd.state in (SDState.DOWN, SDState.INITIALIZING, SDState.DEGRADED):
        add(IssueSeverity.ERROR, "sd_unreadable_state", f"sd is not readable in state {sd.state.value!r}")

    return issues, usable


def _check_plex_sd_layout(
    plex: Plex,
    sds: list[SD],
    volume: bytes | None,
    issues: list[VinumIssue],
) -> bool:
    """Check SD ranges within one plex. Return ``True`` if the layout is usable."""
    if not sds:
        issues.append(
            VinumIssue(
                severity=IssueSeverity.ERROR,
                code="plex_no_sd",
                message="plex has no sd configuration",
                source=plex.source,
                volume=volume,
                plex=plex.name,
            )
        )
        return False

    usable = True

    if plex.org == PlexOrg.CONCAT:
        ordered = sorted(sds, key=lambda sd: (sd.plexoffset if sd.plexoffset is not None else -1, sd.name or b""))
        previous_end: int | None = None
        for sd in ordered:
            if sd.plexoffset is None or sd.length is None:
                usable = False
                continue
            end = sd.plexoffset + sd.length
            if previous_end is None:
                if sd.plexoffset > 0:
                    issues.append(
                        VinumIssue(
                            severity=IssueSeverity.WARNING,
                            code="sd_plexoffset_gap",
                            message=f"first sd starts at {sd.plexoffset:#x} instead of 0",
                            source=sd.source,
                            volume=volume,
                            plex=plex.name,
                            sd=sd.name,
                            drive=sd.drive,
                        )
                    )
            elif sd.plexoffset < previous_end:
                issues.append(
                    VinumIssue(
                        severity=IssueSeverity.ERROR,
                        code="sd_plexoffset_overlap",
                        message=(
                            f"sd range {sd.plexoffset:#x}-{end:#x} overlaps the previous "
                            f"sd ending at {previous_end:#x}"
                        ),
                        source=sd.source,
                        volume=volume,
                        plex=plex.name,
                        sd=sd.name,
                        drive=sd.drive,
                    )
                )
                usable = False
            elif sd.plexoffset > previous_end:
                issues.append(
                    VinumIssue(
                        severity=IssueSeverity.WARNING,
                        code="sd_plexoffset_gap",
                        message=f"sd range starts at {sd.plexoffset:#x}, leaving a gap after {previous_end:#x}",
                        source=sd.source,
                        volume=volume,
                        plex=plex.name,
                        sd=sd.name,
                        drive=sd.drive,
                    )
                )
            previous_end = end

    elif plex.org in (PlexOrg.STRIPED, PlexOrg.RAID5):
        # Striped layouts reuse the plex address space by design. The kernel
        # requires equally sized members with unique, stripe aligned offsets.
        lengths = {sd.length for sd in sds if sd.length is not None}
        if len(lengths) > 1:
            issues.append(
                VinumIssue(
                    severity=IssueSeverity.ERROR,
                    code="sd_length_mismatch",
                    message="striped plex members must have equal lengths",
                    source=plex.source,
                    volume=volume,
                    plex=plex.name,
                )
            )
            usable = False

        seen_offsets: dict[int, SD] = {}
        for sd in sds:
            if sd.plexoffset is None:
                usable = False
                continue
            if sd.plexoffset in seen_offsets:
                issues.append(
                    VinumIssue(
                        severity=IssueSeverity.ERROR,
                        code="sd_plexoffset_duplicate",
                        message=f"duplicate plexoffset {sd.plexoffset:#x} in {plex.org.value!r} plex",
                        source=sd.source,
                        volume=volume,
                        plex=plex.name,
                        sd=sd.name,
                        drive=sd.drive,
                    )
                )
                usable = False
            seen_offsets[sd.plexoffset] = sd

            if plex.stripesize and sd.plexoffset % plex.stripesize:
                issues.append(
                    VinumIssue(
                        severity=IssueSeverity.WARNING,
                        code="sd_plexoffset_unaligned",
                        message=(
                            f"plexoffset {sd.plexoffset:#x} is not a multiple of stripesize {plex.stripesize:#x}"
                        ),
                        source=sd.source,
                        volume=volume,
                        plex=plex.name,
                        sd=sd.name,
                        drive=sd.drive,
                    )
                )

    if plex.org == PlexOrg.RAID5 and len(sds) < 2:
        issues.append(
            VinumIssue(
                severity=IssueSeverity.ERROR,
                code="plex_too_few_sds",
                message="raid5 plex needs at least two members",
                source=plex.source,
                volume=volume,
                plex=plex.name,
            )
        )
        usable = False

    return usable


def analyze_vinum_configs(
    configs: ParsedVinumConfig | VinumConfigs | Iterable[ParsedVinumConfig | VinumConfigs],
    *,
    drives: Iterable[bytes] | None = None,
) -> VinumDiagnostics:
    """Check a Vinum configuration graph without building any virtual disk.

    The analyzer resolves explicit and implicit ``volume -> plex -> sd`` links,
    validates states, lengths, offsets and overlapping ranges, and reports
    problems together with the original configuration line and object names.
    Volumes are assessed independently, so a broken volume never prevents
    another volume from being reported as readable.

    Args:
        configs: One or more parsed configuration blocks. The normal use is to
            pass the blocks of all available physical disks, so duplicate
            definitions can be merged by timestamp.
        drives: Names of the physical drives that are available. When omitted,
            drive availability is not checked.
    """
    if isinstance(configs, ParsedVinumConfig | dict):
        configs = [configs]

    blocks = list(configs)
    diagnostics = VinumDiagnostics()

    volumes: list[Volume] = []
    plexes: list[Plex] = []
    sds: list[SD] = []
    for block in blocks:
        if isinstance(block, ParsedVinumConfig):
            volumes.extend(block.volumes)
            plexes.extend(block.plexes)
            sds.extend(block.sds)
            diagnostics.issues.extend(block.issues)
        else:
            volumes.extend(block["volumes"])
            plexes.extend(block["plexes"])
            sds.extend(block["sds"])

    merged_volumes, conflicted_volumes = _merge_items("volumes", volumes, diagnostics.issues)
    merged_plexes, conflicted_plexes = _merge_items("plexes", plexes, diagnostics.issues)
    merged_sds, conflicted_sds = _merge_items("sds", sds, diagnostics.issues)

    # Nameless volume and plex objects can not be referenced and would otherwise
    # disappear from the report. Keep them visible as configuration evidence.
    for nameless_item, kind in (
        *((item, "volumes") for item in merged_volumes.values()),
        *((item, "plexes") for item in merged_plexes.values()),
    ):
        if nameless_item.name is not None:
            continue
        diagnostics.issues.append(
            VinumIssue(
                severity=IssueSeverity.ERROR,
                code=f"{_singular(kind)}_missing_name",
                message=f"{_singular(kind)} has no name and can not be referenced",
                source=nameless_item.source,
                **_item_objects(kind, nameless_item),
            )
        )

    volume_map: dict[bytes, Volume] = {name: volume for name, volume in merged_volumes.items() if name is not None}
    plex_map: dict[bytes, Plex] = {name: plex for name, plex in merged_plexes.items() if name is not None}

    drive_set = set(drives) if drives is not None else None

    diagnostics.merged = {
        "volumes": volume_map,
        "plexes": plex_map,
        "sds": {name: sd for name, sd in merged_sds.items() if name is not None},
    }

    # Resolve every plex to the volume it belongs to and collect its problems.
    plex_volumes: dict[bytes, bytes | None] = {}
    plex_usable: dict[bytes, bool] = {}
    plex_issues: dict[bytes, list[VinumIssue]] = {}
    mismatched_plex_volumes: dict[bytes, set[bytes]] = defaultdict(set)
    for name, plex in plex_map.items():
        implied_volume = _implied_volume(name)
        volume = plex.volume if plex.volume is not None else implied_volume
        plex_volumes[name] = volume

        issues, usable = _plex_problem_issues(plex, volume, implied_volume, volume_map, name in conflicted_plexes)
        if any(issue.code == "plex_volume_ownership_mismatch" for issue in issues) and implied_volume is not None:
            mismatched_plex_volumes[implied_volume].add(name)
        if plex.volume is None and implied_volume is not None:
            issues.insert(
                0,
                VinumIssue(
                    severity=IssueSeverity.INFO,
                    code="implicit_plex_volume",
                    message=f"plex implicitly associated with volume {_display(implied_volume)} by name",
                    source=plex.source,
                    volume=implied_volume,
                    plex=name,
                ),
            )

        plex_issues[name] = issues
        plex_usable[name] = usable
        diagnostics.issues.extend(issues)

    # Resolve every sd to its plex and collect its problems.
    sds_by_plex: dict[bytes | None, list[SD]] = defaultdict(list)
    sd_usable: dict[int, bool] = {}
    for sd in merged_sds.values():
        implied_plex = _implied_plex(sd.name)
        plex_name = sd.plex if sd.plex is not None else implied_plex
        plex = plex_map.get(plex_name) if plex_name is not None else None
        volume = plex_volumes.get(plex_name) if plex is not None else None
        if volume is None and implied_plex is not None:
            # Keep the broken reference attributable to the volume encoded in
            # the object name, even when the plex itself is missing.
            volume = _implied_volume(implied_plex)
        issues, usable = _sd_problem_issues(
            sd, plex_name, implied_plex, volume, plex_map, drive_set, sd.name in conflicted_sds
        )
        if sd.plex is None and implied_plex is not None:
            issues.insert(
                0,
                VinumIssue(
                    severity=IssueSeverity.INFO,
                    code="implicit_sd_plex",
                    message=f"sd implicitly associated with plex {_display(implied_plex)} by name",
                    source=sd.source,
                    volume=volume,
                    plex=implied_plex,
                    sd=sd.name,
                    drive=sd.drive,
                ),
            )

        diagnostics.issues.extend(issues)
        sd_usable[id(sd)] = usable
        sds_by_plex[plex_name].append(sd)

    # Check the SD layout within each plex, including plexes without any SD.
    for plex_name, plex in plex_map.items():
        plex_sds = sds_by_plex.get(plex_name, [])
        layout_usable = _check_plex_sd_layout(plex, plex_sds, plex_volumes.get(plex_name), diagnostics.issues)
        if not layout_usable or any(not sd_usable.get(id(sd), False) for sd in plex_sds):
            plex_usable[plex_name] = False

    # Check physical drive ranges. Only compare plexes that are otherwise usable,
    # so a broken reference can not poison an independent volume.
    if drive_set is not None:
        ranges_by_drive: dict[bytes, list[tuple[SD, int, int]]] = defaultdict(list)
        for plex_name, plex_sds in sds_by_plex.items():
            if not plex_usable.get(plex_name, False):
                continue
            for sd in plex_sds:
                if not sd_usable.get(id(sd), False) or sd.length is None:
                    continue
                start = sd.driveoffset if sd.driveoffset is not None else 0
                ranges_by_drive[sd.drive].append((sd, start, start + sd.length))

        # Map every SD to the plex and volume it was accepted into, so an
        # unrelated volume is not blamed for another volume's range.
        sd_owners: dict[int, tuple[bytes | None, bytes | None]] = {}
        for plex_name, plex_sds in sds_by_plex.items():
            if not plex_usable.get(plex_name, False):
                continue
            for sd in plex_sds:
                sd_owners[id(sd)] = (plex_name, plex_volumes.get(plex_name))

        for drive, ranges in ranges_by_drive.items():
            ranges.sort(key=lambda entry: (entry[1], entry[2]))
            for idx in range(1, len(ranges)):
                sd, start, end = ranges[idx]
                prev_sd, prev_start, prev_end = ranges[idx - 1]
                if start >= prev_end:
                    continue

                sd_plex, sd_volume = sd_owners.get(id(sd), (sd.plex, None))
                prev_plex, prev_volume = sd_owners.get(id(prev_sd), (prev_sd.plex, None))

                for current, other, plex_name, volume, other_start, other_end in (
                    (sd, prev_sd, sd_plex, sd_volume, prev_start, prev_end),
                    (prev_sd, sd, prev_plex, prev_volume, start, end),
                ):
                    diagnostics.issues.append(
                        VinumIssue(
                            severity=IssueSeverity.ERROR,
                            code="sd_driveoffset_overlap",
                            message=(
                                f"physical range {other_start:#x}-{other_end:#x} on drive {_display(drive)} "
                                f"overlaps sd {_display(other.name)}"
                            ),
                            source=current.source,
                            volume=volume,
                            plex=plex_name,
                            sd=current.name,
                            drive=drive,
                        )
                    )

                # Overlapping physical ranges make both plexes untrustworthy.
                for plex_name in (sd_plex, prev_plex):
                    if plex_name is not None:
                        plex_usable[plex_name] = False

    # Finally assess every volume independently of the others.
    by_volume: dict[bytes, list[Plex]] = defaultdict(list)
    for name, plex in plex_map.items():
        volume = plex_volumes.get(name)
        if volume is not None:
            by_volume[volume].append(plex)

    for name, volume in volume_map.items():
        assessment = VolumeAssessment(volume=volume)

        # Some conditions make a volume evidence-only no matter how healthy its
        # remaining plexes look. Remember them so no later check can promote it.
        forced_evidence = False

        down = volume.state is VolumeState.DOWN

        if name in conflicted_volumes:
            assessment.status = VolumeAvailability.EVIDENCE_ONLY
            assessment.reason = "volume name has conflicting definitions"
            forced_evidence = True

        if name in mismatched_plex_volumes:
            assessment.status = VolumeAvailability.EVIDENCE_ONLY
            assessment.reason = assessment.reason or "plex names target this volume but claim another"
            forced_evidence = True
            assessment.issues.append(
                VinumIssue(
                    severity=IssueSeverity.ERROR,
                    code="plex_volume_ownership_mismatch",
                    message="a plex named for this volume claims another volume",
                    source=volume.source,
                    volume=name,
                    plex=next(iter(mismatched_plex_volumes[name])),
                )
            )

        if down:
            assessment.status = VolumeAvailability.EVIDENCE_ONLY
            assessment.reason = assessment.reason or "volume is down"
            forced_evidence = True
        elif volume.state is None:
            issue = VinumIssue(
                severity=IssueSeverity.WARNING,
                code="volume_state_unknown",
                message="volume has no state, assuming it is usable",
                source=volume.source,
                volume=name,
            )
            assessment.issues.append(issue)
            diagnostics.issues.append(issue)

        usable_plexes = []
        for plex in by_volume.get(name, []):
            assessment.issues.extend(plex_issues.get(plex.name, []))
            if plex_usable.get(plex.name, False):
                usable_plexes.append(plex)

        for issue in diagnostics.issues:
            if issue.volume == name and all(issue is not existing for existing in assessment.issues):
                assessment.issues.append(issue)

        if not by_volume.get(name):
            assessment.status = VolumeAvailability.EVIDENCE_ONLY
            assessment.reason = assessment.reason or "volume has no plex configuration"
        elif forced_evidence:
            pass
        elif not usable_plexes:
            assessment.status = VolumeAvailability.EVIDENCE_ONLY
            assessment.reason = assessment.reason or "volume has no complete readable plex"
        else:
            # A broken plex does not make the volume unreadable: the remaining
            # complete plexes still describe valid data. Keep the loss visible,
            # but do not hide a recoverable mirror or concatenation.
            for plex in by_volume[name]:
                if plex in usable_plexes:
                    continue
                issue = VinumIssue(
                    severity=IssueSeverity.WARNING,
                    code="plex_unusable",
                    message="plex is not usable, continuing with the remaining plexes",
                    source=plex.source,
                    volume=name,
                    plex=plex.name,
                )
                diagnostics.issues.append(issue)
                assessment.issues.append(issue)

            # A mirror needs every plex to describe the same address space.
            if len({plex.org for plex in usable_plexes}) > 1:
                assessment.status = VolumeAvailability.EVIDENCE_ONLY
                assessment.reason = "mirror plexes use different organisations"
                assessment.issues.append(
                    VinumIssue(
                        severity=IssueSeverity.ERROR,
                        code="mixed_plex_org",
                        message="mirror plexes use different organisations",
                        source=usable_plexes[0].source,
                        volume=name,
                        plex=usable_plexes[0].name,
                    )
                )
            elif len({_plex_size(plex, sds_by_plex[plex.name]) for plex in usable_plexes}) > 1:
                assessment.status = VolumeAvailability.EVIDENCE_ONLY
                assessment.reason = "mirror plexes have different sizes"
                assessment.issues.append(
                    VinumIssue(
                        severity=IssueSeverity.ERROR,
                        code="mirror_plex_size_mismatch",
                        message="mirror plexes have different sizes",
                        source=usable_plexes[0].source,
                        volume=name,
                        plex=usable_plexes[0].name,
                    )
                )
            else:
                assessment.status = VolumeAvailability.READABLE
                assessment.reason = None
                assessment.plexes = sorted(usable_plexes, key=lambda plex: plex.name or b"")
                assessment.sds = {plex.name: sds_by_plex[plex.name] for plex in usable_plexes}

        diagnostics.volumes.append(assessment)

    diagnostics.volumes.sort(key=lambda item: item.volume.source.line if item.volume.source else 0)

    return diagnostics
