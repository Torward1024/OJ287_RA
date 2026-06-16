import ctypes
import os
import numpy as np
from typing import NamedTuple
from typing import Any
from enum import Enum
import warnings


UVX_MAX_STOKES = 8
UVX_MAX_TABLES = 128
UVX_ID_FILE = 1003


# ------------------------------------------------------------
# 1.  UVX header (first 512 bytes of file)
# ------------------------------------------------------------
class UVXHeader(ctypes.Structure): # Or ctypes.LittleEndianStructure
    _pack_   = 1               # no padding
    _fields_ = [
        ("id",          ctypes.c_uint32),         # 0
        ("ofs_data",    ctypes.c_uint32),         # 4
        ("ofs_table",   ctypes.c_uint64),         # 8
        ("n_records",   ctypes.c_uint32),         # 16
        ("epoch",       ctypes.c_double),         # 20
        ("date_time",   ctypes.c_double),         # 28
        ("time_sys",    ctypes.c_uint16),         # 36
        ("unit",        ctypes.c_char * 10),      # 38
        ("observer",    ctypes.c_char * 16),      # 48
        ("instrument",  ctypes.c_char * 16),      # 64
        ("u_min",       ctypes.c_double),         # 80
        ("u_max",       ctypes.c_double),         # 88
        ("v_min",       ctypes.c_double),         # 96
        ("v_max",       ctypes.c_double),         # 104
        ("w_min",       ctypes.c_double),         # 112
        ("w_max",       ctypes.c_double),         # 120
        ("t_min",       ctypes.c_double),         # 128
        ("t_max",       ctypes.c_double),         # 136
        ("re_min",      ctypes.c_double),         # 144
        ("re_max",      ctypes.c_double),         # 152
        ("im_min",      ctypes.c_double),         # 160
        ("im_max",      ctypes.c_double),         # 168
        ("weight_min",  ctypes.c_double),         # 176
        ("weight_max",  ctypes.c_double),         # 184
        ("n_stokes",    ctypes.c_uint16),         # 192
        ("m_stokes",    ctypes.c_int16 * UVX_MAX_STOKES), # 194
        ("n_ifs",       ctypes.c_uint32),         # 210
        ("n_channels",  ctypes.c_uint32),         # 214
        ("freq0",       ctypes.c_double),         # 218
        ("dt",          ctypes.c_double),         # 226
        ("freqPix",     ctypes.c_double),         # 234
        ("version",     ctypes.c_double),         # 242
        # /* yyyy-mm-dd */
        ("version",     ctypes.c_char * 12),      # 250
        ("reserved",    ctypes.c_char * 220),     # 262 … 481
    ]
#    HEADER_SIZE = ctypes.sizeof(UVXHeader)        # 482

# ------------------------------------------------------------
# 2.  Visibility record header (fixed 28 bytes)
# ------------------------------------------------------------
UVXRecHdrDTYPE = np.dtype([
    ("uv_flag",   np.uint8),
    ("freq_sel",  np.uint8),
    ("source_no", np.uint16),
    ("u_wave",    np.float32),
    ("v_wave",    np.float32),
    ("w_wave",    np.float32),
    ("time",      np.float64),
    ("tlsc1",     np.uint8),
    ("tlsc2",     np.uint8),
], align=False)
# ------------------------------------------------------------
# 3.  Complex visibility triple (re, im, weight)
# ------------------------------------------------------------
UVXComplexDTYPE = np.dtype([
    ("re", ctypes.c_float),
    ("im", ctypes.c_float),
    ("wt", ctypes.c_float)
], align=False)

# -----------------------------------------------------------
# 1. Directory entry for one table
# -----------------------------------------------------------
class AUVXTableItem(ctypes.Structure):
    _pack_   = 1
    _fields_ = [
        ("id",      ctypes.c_int32),      # table identifier
        ("ofs",     ctypes.c_int64),      # file offset to table start
        ("size",    ctypes.c_int32),      # table size in bytes
    ]

# -----------------------------------------------------------
# 2. Directory header (immediately after last visibility)
# -----------------------------------------------------------
class AUVXTableDirectory(ctypes.Structure):
    _pack_   = 1
    _fields_ = [
        ("ntables", ctypes.c_int16),                           # how many tables
        ("items",   AUVXTableItem * UVX_MAX_TABLES),           # up to 128 entries
    ]

# -----------------------------------------------------------
# 3. Key-value pair inside a table header
# -----------------------------------------------------------
class AUVXTableKey(ctypes.Structure):
    _pack_   = 1
    _fields_ = [
        ("keyName", ctypes.c_char * 10),   # keyword name
        ("keyVal",  ctypes.c_char * 32),   # keyword value
        ("isVal",   ctypes.c_int32),       # 1 = numeric, 0 = string
    ]

# -----------------------------------------------------------
# 4. Column descriptor
# -----------------------------------------------------------
class AUVXTableManual(ctypes.Structure):
    _pack_   = 1
    _fields_ = [
        ("name",   ctypes.c_char * 18),    # column name
        ("unit",   ctypes.c_char * 10),    # physical unit
        ("format", ctypes.c_char * 8),     # FORTRAN-style format
    ]

# -----------------------------------------------------------
# 5. Full table header (per table)
# -----------------------------------------------------------
class AUVXTableHeader(ctypes.Structure):
    _pack_   = 1
    _fields_ = [
        ("id",        ctypes.c_int32),                    # same as AUVXTableItem.id
        ("nkey",      ctypes.c_uint16),                   # number of key-value pairs
        ("keys",      AUVXTableKey * 0),                  # placeholder; actual length = nkey
        # --- immediately follows:
        # rows, columns, columnSize, nmanual
        # AUVXTableManual[nmanual]
        # raw data bytes
    ]
    # NOTE: this is variable-length; use ctypes string buffer
    #       or memoryview after reading nkey / rows / columns / nmanual.

    
class AUVXTableKeyWordHeader(ctypes.Structure):
    _pack_   = 1
    _fields_ = [
        ("m_id",        ctypes.c_int32), # table id
        ("nkeywords",   ctypes.c_int16), # table keyword number
    ]
    
class AUVXTableManualHeader(ctypes.Structure):
    _pack_   = 1
    _fields_ = [
        ("m_rows",       ctypes.c_int32),
        ("m_columns" ,   ctypes.c_int32),
	    ("m_columnSize", ctypes.c_int32),
    ]

# ---------- helper ----------
def data_size_bytes(header) -> int:
    """Total payload size for all visibility triplets."""
    return (
        header.n_records
        * header.n_ifs
        * header.n_channels
        * header.n_stokes
        * 3 * ctypes.sizeof(ctypes.c_float)
    )


def _desc2type(format: str):
    if len(format) == 1:
        size = 1
        ftype = format[0]
    else:
        size = int(format[:-1:])
        ftype = format[-1]
    dtype: Any
    match ftype:
        case "L":
            dtype = np.uint8
        case "J":
            dtype = np.int32
        case "I":
            dtype = np.int16
        case "A":
            return (np.dtype(f"S{size}"),)
        case "E":
            dtype = np.float32
        case "D":
            dtype = np.float64
        case _:
            raise RuntimeError(f"unknown type {ftype}")
    if size == 1:
        return (dtype,)
    else:
        return dtype, (size,)


class _UVXTab(NamedTuple):
    attrs: dict[str, str | float]
    data: np.ndarray


class VLBIPolarization(Enum):
    RR = -1
    LL = -2
    RL = -3
    LR = -4
    I = 1
    Q = 2
    U = 3
    V = 4


class UVXReader(object):
    def __init__(self, path: os.PathLike, flags="rb"):
        self.path = path
        self._file = None
        self._hdr = None
        self._ofs_data = None
        self._vistype = None
        self._flags = flags
        self.tables: dict[str, _UVXTab] = {}

    @property
    def hdr(self) -> UVXHeader:
        assert self._hdr is not None
        return self._hdr

    @property
    def stokes(self) -> list[VLBIPolarization]:
        assert self._hdr is not None
        hrd = self._hdr
        result = [
            VLBIPolarization(pol)
            for pol in self._hdr.m_stokes[:self._hdr.n_stokes:]
        ]
        return result

    def __getitem__(self, key: slice):
        if isinstance(key, slice):
            return self._getslice(key)
        else:
            raise RuntimeError(
                f"type {type(key)} is unsupported"
            )

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, type, value, tb):
        self._close()

    def __len__(self):
        return int(self._hdr.n_records)

    def _getslice(self, key: slice) -> np.ndarray:
        assert key.start is not None
        assert key.start >= 0, "negative slice start is not supported yet"
        assert key.stop is not None, "slice stop should be provided"
        assert key.stop >= 0, "negative slice stop is not supported yet"
        assert key.step is None or key.step == 1

        stop = min(key.stop, len(self))
        assert 0 <= key.start < len(self)
        count = max(0, key.stop - key.start)

        assert self._file is not None
        self._file.seek(self._ofs_data + key.start * self._vistype.itemsize,
                        os.SEEK_SET)
        return np.fromfile(
            self._file, dtype=self._vistype, count=count, offset=0
        )

    def _setslice(self, key: slice, data: np.ndarray) -> None:
        before = self._getslice(key)
        after = data
        assert before.shape == after.shape
        header_before = before["header"]
        header_after = after["header"]
        complx_before = before["complex"]
        complx_after = after["complex"]
        assert complx_before.shape == complx_after.shape
        assert np.logical_and(
            header_before["uv_flag"],
            np.logical_not(header_after["uv_flag"])
        ).sum() == 0
        for hkey in ["freq_sel", "source_no", "u_wave", "v_wave", "w_wave",
                    "time", "tlsc1", "tlsc2"]:
            assert np.isclose(header_before[hkey],
                              header_after[hkey]).all(), f"{hkey}"
        assert self._file is not None
        self._file.seek(self._ofs_data + key.start * self._vistype.itemsize,
                        os.SEEK_SET)
        after.tofile(self._file)

    def _open(self):

        self._file = open(self.path, self._flags)

        self._hdr = UVXHeader.from_buffer_copy(
            self._file.read(ctypes.sizeof(UVXHeader))
        )

        sz = data_size_bytes(self._hdr)

        assert sz > 0, "there is no vis data"

        self._ofs_data = self._hdr.ofs_data

        self._vistype = np.dtype([
            ("header", UVXRecHdrDTYPE),
            ("complex", UVXComplexDTYPE, (
                self._hdr.n_ifs,
                self._hdr.n_channels,
                self._hdr.n_stokes
            ))
        ], align=False)

        # ----------------------------------------
        # Нам нужны только visibility records.
        # Таблицы UVX (AN/FQ/SU/etc.) не читаем.
        # ----------------------------------------

        self.tables = {}

        return

    def _close(self):
        self._hdr = None
        self._ofs_data = None
        self._vistype = None
        self.tables = {}
        self._file.close()
        self._file = None

