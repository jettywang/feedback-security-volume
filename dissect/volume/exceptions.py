from __future__ import annotations


class Error(Exception):
    pass


class LVM2Error(Error):
    pass


class DiskError(Error):
    pass


class GPTConflictError(DiskError):
    """The primary and backup GPT copies are both valid, but describe different layouts.

    Both copies remain available on the exception so the conflicting evidence can be
    inspected. The original image is never modified.

    Args:
        primary: Validation result of the primary GPT copy.
        backup: Validation result of the backup GPT copy.
        conflicts: Human readable descriptions of the differences.
    """

    def __init__(self, primary: object, backup: object, conflicts: list[str]):
        self.primary = primary
        self.backup = backup
        self.conflicts = list(conflicts)
        super().__init__(
            "GPT primary and backup copies are both valid but conflict; "
            "an explicit source must be selected:\n- " + "\n- ".join(self.conflicts)
        )


class RAIDError(Error):
    pass


class DMError(Error):
    pass


class MDError(Error):
    pass


class DDFError(Error):
    pass
