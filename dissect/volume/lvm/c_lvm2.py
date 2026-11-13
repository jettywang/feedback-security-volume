from __future__ import annotations

from dissect.cstruct import cstruct

lvm_def = """
struct label_header {
    char    id[8];
    uint64  sector;
    uint32  crc;
    uint32  offset;
    char    type[8];
};

struct pv_header {
    char    pv_uuid[32];
    uint64  device_size;
};

struct disk_locn {
    uint64  offset;         /* Offset in bytes to start sector */
    uint64  size;           /* Bytes */
};

// Metadata area header
struct mda_header {
    uint32  checksum;       /* Checksum of rest of mda_header */
    char    magic[16];      /* To aid scans for metadata */
    uint32  version;
    uint64  start;          /* Absolute start byte of mda_header */
    uint64  size;           /* Size of metadata area */
};

struct raw_locn {
    uint64  offset;         /* Offset in bytes to start sector */
    uint64  size;           /* Bytes */
    uint32  checksum;
    uint32  flags;
};

#define RAW_LOCN_IGNORED    0x00000001
"""

c_lvm = cstruct().load(lvm_def)

SECTOR_SIZE = 512

LABEL_SCAN_SECTORS = 4

# LVM2 uses the regular (reflected) CRC32 polynomial, but seeds the calculation with a fixed value and does not
# XOR the result with 0xffffffff afterwards. This is the implementation from lvm2's lib/misc/crc.c.
CRC32_POLYNOMIAL = 0xEDB88320
CRC32_INIT_SEED = 0xF597A6CF

# Precompute the reflected CRC32 lookup table once.
_CRC32_TABLE = []
for _i in range(256):
    _crc = _i
    for _ in range(8):
        _crc = (_crc >> 1) ^ CRC32_POLYNOMIAL if _crc & 1 else _crc >> 1
    _CRC32_TABLE.append(_crc)


def lvm_crc32(data: bytes, seed: int = CRC32_INIT_SEED) -> int:
    """Compute the CRC32 variant used by LVM2 to protect labels, metadata area headers and metadata text.

    Unlike :func:`zlib.crc32`, the LVM2 checksum starts from a fixed :data:`CRC32_INIT_SEED` and the resulting
    value is not XORed with ``0xffffffff``.

    Args:
        data: The bytes to calculate the checksum over.
        seed: The initial CRC value. Defaults to the LVM2 seed.

    Returns:
        The 32-bit checksum as an unsigned integer.
    """
    crc = seed
    for byte in data:
        crc = _CRC32_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc & 0xFFFFFFFF


STATUS_FLAG_ALLOCATABLE = "ALLOCATABLE"  # pv only
STATUS_FLAG_RESIZEABLE = "RESIZEABLE"  # vg only
STATUS_FLAG_READ = "READ"
STATUS_FLAG_VISIBLE = "VISIBLE"  # lv only
STATUS_FLAG_WRITE = "WRITE"
