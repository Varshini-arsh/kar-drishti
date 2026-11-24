"""Facts about this computer that Settings shows: free and total memory.

Standard library only (no extra install). Works on Windows and Linux;
returns None where the number cannot be read.
"""
from __future__ import annotations

import ctypes
import os
import sys


def memory_gb() -> tuple[float | None, float | None]:
    """(free GB, total GB) of RAM, rounded to 0.1."""
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullAvailPhys / 2**30, 1), round(st.ullTotalPhys / 2**30, 1)
        if os.path.exists("/proc/meminfo"):
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k] = int(v.split()[0]) * 1024
            return round(info["MemAvailable"] / 2**30, 1), round(info["MemTotal"] / 2**30, 1)
    except Exception:
        pass
    return None, None
