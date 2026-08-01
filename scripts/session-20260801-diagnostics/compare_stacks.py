"""Compare old-path vs native-path stacks: sharpness, banding, crops."""
import os, sys, struct, zlib
import numpy as np

def read_fits_f32(path):
    with open(path, 'rb') as f:
        hdr = b''
        while True:
            block = f.read(2880)
            hdr += block
            if b'END     ' in block:
                break
        cards = {}
        for i in range(0, len(hdr), 80):
            c = hdr[i:i+80].decode('ascii', 'replace')
            if '=' in c:
                k, v = c.split('=', 1)
                cards[k.strip()] = v.split('/')[0].strip()
        w = int(cards['NAXIS1']); h = int(cards['NAXIS2'])
        data = np.frombuffer(f.read(w*h*4), dtype='>f4').reshape(h, w).astype(np.float32)
    return data

def write_png_gray8(path, arr):
    h, w = arr.shape
    raw = b''.join(b'\x00' + arr[i].tobytes() for i in range(h))
    def chunk(tag, data):
        c = tag + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n'
                + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 6))
                + chunk(b'IEND', b''))

def disc_mask(img, frac=0.85):
    thr = np.percentile(img, 75) * 0.4
    ys, xs = np.where(img > thr)
    cx, cy = xs.mean(), ys.mean()
    r = np.sqrt((img > thr).sum() / np.pi)
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    return ((X-cx)**2 + (Y-cy)**2) < (r*frac)**2, (cx, cy, r)

def gauss_blur(img, sigma):
    from math import ceil
    r = int(ceil(3*sigma))
    x = np.arange(-r, r+1)
    k = np.exp(-x**2/(2*sigma**2)); k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), 0, img)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), 1, out)

def metrics(img, name):
    m, (cx, cy, r) = disc_mask(img)
    blur = gauss_blur(img, 2.0)
    hf = (img - blur)[m]
    med = np.median(img[m])
    hf_energy = np.mean(hf**2) / med**2 * 1e4
    gy, gx = np.gradient(img)
    grad = np.sqrt(gx**2+gy**2)[m].mean() / med * 1e3
    # banding: high-pass row-mean profile inside disc
    rows = np.where(m, img, np.nan)
    rowmean = np.nanmean(rows, axis=1)
    ok = ~np.isnan(rowmean)
    rm = rowmean[ok]
    k = 33
    lo = np.convolve(rm, np.ones(k)/k, mode='same')
    band = np.nanstd((rm - lo)[k:-k]) / med * 1e3
    print(f"{name:<28} med={med:9.1f} HFx1e4={hf_energy:8.3f} grad_x1e3={grad:7.3f} band_x1e3={band:6.3f} r={r:.0f} c=({cx:.0f},{cy:.0f})")
    return (cx, cy, r)

def crop_png(img, cx, cy, r, out, which):
    lo, hi = np.percentile(img, [1, 99.8])
    g = np.clip((img-lo)/(hi-lo), 0, 1)
    g8 = (np.sqrt(g)*255).astype(np.uint8)
    if which == 'center':
        x0, y0 = int(cx)-400, int(cy)-400
    elif which == 'limb':
        x0, y0 = int(cx)-400, int(cy-r*0.92)-100
    else:  # activeregion: brightest HF area — just west of center
        x0, y0 = int(cx)-800, int(cy)-100
    x0 = max(x0, 0); y0 = max(y0, 0)
    write_png_gray8(out, g8[y0:y0+800:1, x0:x0+800:1])

if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)) + r'\realstack'
    pairs = [('native8.fits', 'NATIVE stack'), ('oldstack.fits', 'OLD-path stack'),
             ('native8-scan01.fits', 'single scan 01')]
    geo = {}
    for fn, label in pairs:
        p = os.path.join(base, fn)
        if os.path.exists(p):
            img = read_fits_f32(p)
            geo[fn] = (img, metrics(img, label))
    for fn, label in pairs:
        if fn in geo:
            img, (cx, cy, r) = geo[fn]
            stem = fn.replace('.fits', '')
            for which in ('center', 'limb'):
                crop_png(img, cx, cy, r, os.path.join(base, f'{stem}_{which}.png'), which)
    print('crops written')
