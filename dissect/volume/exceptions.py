from __future__ import annotations


class Error(Exception):
    pass


class LVM2Error(Error):
    pass


class DiskError(Error):
    pass


class RAIDError(Error):
    pass


class RAIDStripeReadError(RAIDError):
    """Raised when a RAID stripe cannot be reconstructed from the available members.

    This is never raised for a recoverable single-member failure. It is raised when a stripe has
    too many missing/failed members to rebuild, or when a member that is required for the rebuild
    returns a short read. The failure is reported instead of silently padding with zero bytes.

    Args:
        message: Human readable description of the failure.
        evidence: Per-member :class:`~dissect.volume.raid.stream.MemberEvidence` describing why each
            member involved in the failed interval could not be used.
        logical_offset: The start of the unreadable interval in the logical RAID volume.
        logical_length: The length of the unreadable interval in the logical RAID volume.
        stripe_number: The RAID stripe number that could not be reconstructed.
    """

    def __init__(
        self,
        message: str,
        evidence: list,
        logical_offset: int,
        logical_length: int,
        stripe_number: int,
    ):
        super().__init__(message)
        self.evidence = evidence
        self.logical_offset = logical_offset
        self.logical_length = logical_length
        self.stripe_number = stripe_number


class DMError(Error):
    pass


class MDError(Error):
    pass


class DDFError(Error):
    pass
