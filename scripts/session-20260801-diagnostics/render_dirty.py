"""Render dirty reconstructions to PNG for visual check (no pipeline involved)."""
import os, glob, zlib, struct
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

def write_png_gray8(path, arr):
    h, w = arr.shape
    raw = b''.join(b'\x00' + arr[i].tobytes() for i in range(h))
    def chunk(tag, data):
        c = tag + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 6))
           + chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(png)

for p in sorted(glob.glob(os.path.join(OUT, 'dirty_scan-0[12].ser.npy'))):
    name = os.path.basename(p).replace('dirty_', '').replace('.ser.npy', '')
    ha = np.load(p).astype(np.float32)  # (frames, 1920) core-row recon, x downsampled 2x
    # downsample frames 2x to roughly square-ish, normalize robustly
    img = ha[::2]
    lo, hi = np.percentile(img, [0.5, 99.7])
    g = np.clip((img - lo) / (hi - lo), 0, 1)
    g = (np.sqrt(g) * 255).astype(np.uint8)   # gamma for limb visibility
    write_png_gray8(os.path.join(OUT, f'{name}_dirty.png'), g)
    print(name, g.shape)

# overlay the measured center trace on scan-01 as a white line
p = os.path.join(OUT, 'resid_scan-01.ser.npy')
d = np.load(p)
fr, centers = d[:, 0].astype(int), d[:, 3]
ha = np.load(os.path.join(OUT, 'dirty_scan-01.ser.npy')).astype(np.float32)
img = ha[::2]
lo, hi = np.percentile(img, [0.5, 99.7])
g = np.clip((img - lo) / (hi - lo), 0, 1)
g = (np.sqrt(g) * 255).astype(np.uint8)
for f_, c_ in zip(fr, centers):
    r = f_ // 2
    cc = int(c_ / 2)
    if 0 <= r < g.shape[0] and 0 <= cc < g.shape[1]:
        g[r, max(cc-2,0):cc+3] = 255
write_png_gray8(os.path.join(OUT, 'scan-01_dirty_centerline.png'), g)
print('centerline overlay written')
