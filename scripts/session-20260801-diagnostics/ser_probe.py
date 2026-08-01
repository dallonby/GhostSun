import struct, sys, os, glob
import numpy as np

def read_ser_header(path):
    with open(path, 'rb') as f:
        h = f.read(178)
    fileid = h[0:14].decode('ascii', 'replace').strip()
    luid, colorid, endian, width, height, depth, frames = struct.unpack_from('<iiiiiii', h, 14)
    observer = h[42:82].decode('ascii', 'replace').strip('\x00 ')
    instrument = h[82:122].decode('ascii', 'replace').strip('\x00 ')
    telescope = h[122:162].decode('ascii', 'replace').strip('\x00 ')
    dt, dt_utc = struct.unpack_from('<qq', h, 162)
    return dict(fileid=fileid, colorid=colorid, width=width, height=height,
                depth=depth, frames=frames, instrument=instrument,
                telescope=telescope, size=os.path.getsize(path))

base = r'C:\Users\djall\Documents\Codex\2026-07-23\i\outputs\GhostSun-ha-lpf-runway-20260731\GhostSun-Windows-x64\GhostSun Captures\scan-1785599603'
for p in sorted(glob.glob(os.path.join(base, 'scan-*.ser'))):
    try:
        h = read_ser_header(p)
        bytes_per_px = 2 if h['depth'] > 8 else 1
        expected = 178 + h['frames'] * h['width'] * h['height'] * bytes_per_px
        trailer = h['size'] - expected  # timestamp trailer is 8 bytes/frame if present
        print(f"{os.path.basename(p)}: {h['width']}x{h['height']} depth={h['depth']} "
              f"frames={h['frames']} trailer={trailer} instr='{h['instrument']}'")
    except Exception as e:
        print(f"{os.path.basename(p)}: ERROR {e}")
