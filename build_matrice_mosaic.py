#!/usr/bin/env python
"""Georeferenced mosaic of DJI Matrice 4T nadir visible photos over Ceva Lake.

Footprint-projects each RTK-geotagged nadir photo onto a UTM-18N grid (center from
EXIF GPS, orientation from XMP gimbal-yaw, scale from relative altitude + lens FOV),
feather-blends overlaps, and writes a GeoTIFF that overlays the hyperspectral mosaics.
NOT a true SfM orthomosaic (no bundle adjustment) but accurate enough for QGIS overlay
given RTK positions + nadir view + flat lake.
"""
import os, re, sys, math, numpy as np
from PIL import Image, ExifTags
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from scipy.ndimage import map_coordinates

UTM_EPSG   = 32618          # match the hyperspectral
RES_M      = 0.10           # output GSD (native ~0.04 m; 0.10 keeps it light)
MAXW       = 1600           # downsample each photo to this width for speed
FF_DIAG_MM = 43.2666        # full-frame diagonal (for 35mm-equiv -> FOV)
OUT        = os.path.expanduser('~/bayspec_results')
D = open('/tmp/matrice_folder.txt').read().strip()
os.makedirs(OUT, exist_ok=True)

TAGS = {v: k for k, v in ExifTags.TAGS.items()}
def to_f(x):
    try: return float(x)
    except Exception: return x[0] / x[1]
def dms(v): return to_f(v[0]) + to_f(v[1]) / 60 + to_f(v[2]) / 3600

def read_meta(path):
    im = Image.open(path); ex = im._getexif() or {}
    t = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
    g = {ExifTags.GPSTAGS.get(k, k): v for k, v in t['GPSInfo'].items()}
    lat = dms(g['GPSLatitude']) * (-1 if g.get('GPSLatitudeRef') == 'S' else 1)
    lon = dms(g['GPSLongitude']) * (-1 if g.get('GPSLongitudeRef') == 'W' else 1)
    foc35 = to_f(t.get('FocalLengthIn35mmFilm', 24))
    raw = open(path, 'rb').read(); xs = raw.find(b'<x:xmpmeta'); xe = raw.find(b'</x:xmpmeta')
    xmp = raw[xs:xe + 12].decode('latin1') if xs != -1 else ''
    def xg(k, d):
        m = re.search(k + r'="?\+?([-\d\.]+)', xmp); return float(m.group(1)) if m else d
    yaw = xg('GimbalYawDegree', 0.0); relalt = xg('RelativeAltitude', 118.0)
    return lat, lon, yaw, relalt, foc35

vis = [f for f in sorted(os.listdir(D)) if f.lower().endswith('.jpg') and '_v' in f.lower()]
print(f'{len(vis)} visible photos')

# ---- per-photo geometry ----
photos = []
lons, lats = [], []
for f in vis:
    lat, lon, yaw, H, foc35 = read_meta(os.path.join(D, f))
    lons.append(lon); lats.append(lat)
    photos.append(dict(f=f, lat=lat, lon=lon, yaw=yaw, H=H, foc35=foc35))
E, N = warp_transform(CRS.from_epsg(4326), CRS.from_epsg(UTM_EPSG), lons, lats)
for p, e, n in zip(photos, E, N): p['E'], p['N'] = e, n

# footprint size from FOV + altitude
for p in photos:
    diagFOV = 2 * math.atan(FF_DIAG_MM / (2 * p['foc35']))
    ground_diag = 2 * p['H'] * math.tan(diagFOV / 2)
    p['gsd'] = ground_diag / math.hypot(4032, 3024)        # native m/px (full res)
    p['half_w_m'] = p['gsd'] * 4032 / 2
    p['half_h_m'] = p['gsd'] * 3024 / 2
print(f"altitude ~{np.mean([p['H'] for p in photos]):.0f} m | native GSD ~{np.mean([p['gsd'] for p in photos])*100:.1f} cm")

# ---- output grid ----
pad = max(max(p['half_w_m'], p['half_h_m']) for p in photos) * 1.45
xmin = min(p['E'] for p in photos) - pad; xmax = max(p['E'] for p in photos) + pad
ymin = min(p['N'] for p in photos) - pad; ymax = max(p['N'] for p in photos) + pad
ncol = int((xmax - xmin) / RES_M); nrow = int((ymax - ymin) / RES_M)
print(f'grid {nrow}x{ncol} @ {RES_M} m  ({(xmax-xmin):.0f} x {(ymax-ymin):.0f} m)')
rgbacc = np.zeros((nrow, ncol, 3), np.float32); wbest = np.zeros((nrow, ncol), np.float32)

# ---- project each photo (only over its grid sub-window) ----
for k, p in enumerate(photos):
    im = Image.open(os.path.join(D, p['f'])).convert('RGB')
    sc = MAXW / im.width; im = im.resize((MAXW, int(im.height * sc)))
    arr = np.asarray(im, np.float32); h, w = arr.shape[:2]
    g = (2 * p['H'] * math.tan(math.atan(FF_DIAG_MM / (2 * p['foc35'])))) / math.hypot(w, h)  # m/px at this res
    th = math.radians(p['yaw']); ct, st = math.cos(th), math.sin(th)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    # grid sub-window covering this footprint
    rad = math.hypot(w, h) / 2 * g * 1.1
    c0 = max(0, int((p['E'] - rad - xmin) / RES_M)); c1 = min(ncol, int((p['E'] + rad - xmin) / RES_M))
    r0 = max(0, int((ymax - (p['N'] + rad)) / RES_M)); r1 = min(nrow, int((ymax - (p['N'] - rad)) / RES_M))
    if c1 <= c0 or r1 <= r0: continue
    cols = xmin + (np.arange(c0, c1) + 0.5) * RES_M
    rows = ymax - (np.arange(r0, r1) + 0.5) * RES_M
    EE, NN = np.meshgrid(cols, rows)
    Eoff = EE - p['E']; Noff = NN - p['N']
    ecam = Eoff * ct - Noff * st                  # rotate into camera frame (yaw cw from north)
    ncam = Eoff * st + Noff * ct
    col_i = cx + ecam / g; row_i = cy - ncam / g
    inside = (col_i >= 0) & (col_i <= w - 1) & (row_i >= 0) & (row_i <= h - 1)
    samp = np.stack([map_coordinates(arr[:, :, b], [row_i, col_i], order=1, mode='constant', cval=np.nan)
                     for b in range(3)], axis=-1)
    # feather weight: 1 at center -> 0 at edges
    wgt = np.clip(1 - np.maximum(np.abs(col_i - cx) / (w / 2), np.abs(row_i - cy) / (h / 2)), 0, 1) ** 2
    wgt = np.where(inside & np.isfinite(samp).all(-1), wgt, 0)
    # BEST-photo-per-pixel: keep the most-centered photo's pixels (sharp, no ghosting)
    subw = wbest[r0:r1, c0:c1]
    sub = rgbacc[r0:r1, c0:c1, :]
    better = wgt > subw
    sub[better] = np.nan_to_num(samp)[better]
    subw[better] = wgt[better]
    print(f'  placed {k+1}/{len(photos)} {p["f"]}')

mos = rgbacc
alpha = (wbest > 0).astype(np.uint8) * 255
rgb8 = np.clip(mos, 0, 255).astype(np.uint8)

tr = from_origin(xmin, ymax, RES_M, RES_M)
tif = os.path.join(OUT, 'ceva_matrice_rgb_utm18n.tif')
with rasterio.open(tif, 'w', driver='GTiff', height=nrow, width=ncol, count=4, dtype='uint8',
                   crs=CRS.from_epsg(UTM_EPSG), transform=tr, photometric='RGB', compress='deflate') as ds:
    for b in range(3): ds.write(rgb8[:, :, b], b + 1)
    ds.write(alpha, 4)
    ds.colorinterp = [rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                      rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha]
print('WROTE', tif)

# preview + overlay-extent comparison with the hyperspectral pass
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ext = [xmin, xmax, ymin, ymax]
plt.figure(figsize=(9, 9))
plt.imshow(np.dstack([rgb8, alpha]), extent=ext, origin='upper')
hyp = os.path.join(OUT, 'ceva_PASS_truecolor_utm18n.tif')
if os.path.exists(hyp):
    with rasterio.open(hyp) as ds:
        b = ds.bounds
    plt.gca().add_patch(plt.Rectangle((b.left, b.bottom), b.right - b.left, b.top - b.bottom,
                        fill=False, ec='red', lw=2, label='hyperspectral pass extent'))
    plt.legend()
plt.title(f'Matrice RGB mosaic ({len(photos)} photos) — UTM18N {RES_M}m')
plt.xlabel('easting'); plt.ylabel('northing'); plt.ticklabel_format(useOffset=False, style='plain')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'ceva_matrice_rgb_preview.png'), dpi=110)
print('DONE')
