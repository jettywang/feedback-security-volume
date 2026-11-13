from __future__ import annotations


class Error(Exception):
    pass


class LVM2Error(Error):
    pass


class DiskError(Error):
    pass


class RAIDError(Error):
    pass


class DMError(Error):
    pass


class MDError(Error):
    pass


class DDFError(Error):
    pass


class DDFLayoutError(DDFError):
    """Raised when the on-disk DDF RAID layout fields describe an unsupported or contradictory configuration.

    The offending on-disk field values are kept on the exception so that they can be inspected when the
    metadata has to be reviewed manually.
    """

    def __init__(
        self,
        message: str,
        primary_raid_level: int,
        raid_level_qualifier: int,
        secondary_raid_level: int,
        primary_element_count: int,
        secondary_element_count: int,
        secondary_element_seq: int | None = None,
    ):
        super().__init__(message)
        self.primary_raid_level = primary_raid_level
        self.raid_level_qualifier = raid_level_qualifier
        self.secondary_raid_level = secondary_raid_level
        self.primary_element_count = primary_element_count
        self.secondary_element_count = secondary_element_count
        self.secondary_element_seq = secondary_element_seq
