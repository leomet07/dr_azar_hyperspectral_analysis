#!/usr/bin/env python
"""Single clean pass -> smooth, glint-free, georeferenced mosaic (true-color + NDCI).

Uses ONLY the selected frames (one straight pass over the lake). Steps:
  1. calibrate to reflectance (DN-dark)/(white-dark)  [removes across-track vignette]
  2. inter-frame normalization (one gain per frame)   [smooth frame-to-frame]
  3. COMPLETE glint removal: mask high-NIR + specular brightness outliers -> NaN
  4. rasterize to UTM 18N, then FILL masked gaps (nearest) + gaussian SMOOTH
  5. write true-color RGB + NDCI GeoTIFFs and PNG previews
"""
import numpy as np, json, datetime, csv, os
import xml.etree.ElementTree as ET
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation

SWATH_M   = 80.0
RES_M     = 0.30
BANDS_NM  = [470, 550, 620, 665, 675, 705, 740, 800]
RGB_NM    = (620, 550, 470)
SAT_BIAS  = 3.5
SMOOTH_SIG = 2.5          # gaussian sigma (grid cells) for final smoothing (stronger)
FILL_RAD   = 6            # max cells to fill across a glint gap / show as footprint
OUT = os.path.expanduser('~/bayspec_results')

sel = json.load(open('/tmp/v4_inputs.json'))
XML, DARK, WHITE = sel['XML'], sel['DARK'], sel['WHITE']
LAKES = sorted(sel['LAKES']); RAW = os.path.dirname(LAKES[0])
LOGF = open('/tmp/flightlog_path.txt').read().strip()

r = ET.parse(XML).getroot(); s = r.find('sensor'); g = lambda t, f=float: f(s.find(t).text)
W, H, LPB, BPX = int(g('width', int)), int(g('height', int)), int(g('linesperband', int)), int(g('bandpixels', int))
nb, ns = H // LPB, W // BPX
by = np.arange(nb) * LPB + LPB / 2
wls = g('c0') + g('c1') * by + g('c2') * by ** 2
bidx = [int(np.argmin(np.abs(wls - x))) for x in BANDS_NM]
def bi(nm): return BANDS_NM.index(min(BANDS_NM, key=lambda x: abs(x - nm)))
iR, iG, iB = bi(620), bi(550), bi(470); i665, i705, i800 = bi(665), bi(705), bi(800)

def cube(path):
    a = np.fromfile(path, dtype=np.uint8)
    if a.size != W * H: return None
    a = a.reshape(H, W).astype(np.float32)
    return a[:nb * LPB, :ns * BPX].reshape(nb, LPB, ns, BPX).mean(axis=(1, 3))
dfull = cube(DARK); wfull = cube(WHITE)
dark, white = dfull[bidx, :], wfull[bidx, :]
denom = np.where(np.abs(white - dark) < 0.5, np.nan, white - dark)
panel = wfull.mean(0) > 100

# ---- read + calibrate the pass ----
N = len(LAKES); B = len(BANDS_NM)
REFL = np.full((N, ns, B), np.nan, np.float32)
for i, p in enumerate(LAKES):
    cu = cube(p)
    if cu is None: continue
    R = (cu[bidx, :] - dark) / denom * 0.99 / SAT_BIAS
    R[:, ~panel] = np.nan
    REFL[i] = R.T
print(f'read {N} frames')

# ---- inter-frame normalization (smooth between frames) ----
fb = np.nanmedian(REFL.reshape(N, -1), axis=1)
gain = np.where(np.isfinite(fb) & (fb > 1e-6), np.nanmedian(fb) / fb, np.nan)
REFL = REFL * gain[:, None, None]

# ---- COMPLETE glint removal ----
nir = REFL[:, :, i800]
vis = np.nanmean(REFL[:, :, [iR, iG, iB]], axis=2)
fin = np.isfinite(nir) & np.isfinite(vis)
nir_thr = np.nanpercentile(nir[fin], 45)                       # keep only darkest-NIR (cleanest water)
med, mad = np.nanmedian(vis[fin]), np.nanmedian(np.abs(vis[fin] - np.nanmedian(vis[fin])))
bright_thr = med + 2 * 1.4826 * mad                           # tighter specular-outlier cutoff
glint = (~fin) | (nir > nir_thr) | (vis > bright_thr)
print(f'glint mask removes {100*glint.mean():.1f}% of pixels (NIR>{nir_thr:.4f} or vis>{bright_thr:.4f})')
REFL[np.broadcast_to(glint[:, :, None], REFL.shape)] = np.nan

# ---- georeference ----
def sod_ampm(t):
    d = datetime.datetime.strptime(t.strip(), '%I:%M:%S.%f %p')
    return d.hour*3600 + d.minute*60 + d.second + d.microsecond/1e6
def sod_fname(n):
    q = os.path.basename(n).split('_')[2].split('-'); h, m, s_, ms = (int(x) for x in q)
    return h*3600 + m*60 + s_ + ms/1000.0
gt, glat, glon, gyaw = [], [], [], []
with open(LOGF, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        try:
            gt.append(sod_ampm(row['CUSTOM.updateTime [local]']))
            glat.append(float(row['OSD.latitude'])); glon.append(float(row['OSD.longitude']))
            gyaw.append(float(row['OSD.yaw [360]']))
        except Exception: pass
gt = np.array(gt); o = np.argsort(gt)
gt = gt[o]; glat = np.array(glat)[o]; glon = np.array(glon)[o]; gyaw = np.deg2rad(np.array(gyaw)[o])
ft = np.array([sod_fname(p) for p in LAKES])
flat = np.interp(ft, gt, glat); flon = np.interp(ft, gt, glon)
fyaw = np.arctan2(np.interp(ft, gt, np.sin(gyaw)), np.interp(ft, gt, np.cos(gyaw)))
cE, cN = warp_transform(CRS.from_epsg(4326), CRS.from_epsg(32618), list(flon), list(flat))
cE = np.array(cE); cN = np.array(cN)
j = np.arange(ns); off = (j - (ns - 1) / 2.0) * (SWATH_M / ns)
bear = fyaw[:, None] + np.pi / 2
sE = cE[:, None] + off[None, :] * np.sin(bear)
sN = cN[:, None] + off[None, :] * np.cos(bear)

valid = np.isfinite(REFL).any(2) & np.isfinite(sE) & np.isfinite(sN)
xmin, xmax = sE[valid].min() - 1, sE[valid].max() + 1
ymin, ymax = sN[valid].min() - 1, sN[valid].max() + 1
ncol = int(np.ceil((xmax - xmin) / RES_M)); nrow = int(np.ceil((ymax - ymin) / RES_M))
col = np.clip(((sE - xmin) / RES_M).astype(int), 0, ncol - 1)
rowi = np.clip(((ymax - sN) / RES_M).astype(int), 0, nrow - 1)
tr = from_origin(xmin, ymax, RES_M, RES_M)
print(f'grid {nrow}x{ncol} @ {RES_M}m')

def grid_layers(values):
    L = values.shape[2]; out = np.zeros((L, nrow, ncol)); cnt = np.zeros((L, nrow, ncol))
    rr = np.broadcast_to(rowi[:, :, None], values.shape); cc = np.broadcast_to(col[:, :, None], values.shape)
    ok = np.isfinite(values)
    for l in range(L):
        m = ok[:, :, l]
        np.add.at(out[l], (rr[:, :, l][m], cc[:, :, l][m]), values[:, :, l][m])
        np.add.at(cnt[l], (rr[:, :, l][m], cc[:, :, l][m]), 1)
    with np.errstate(invalid='ignore'):
        return np.where(cnt > 0, out / cnt, np.nan)

def fill_smooth(layer):
    """nearest-fill small gaps, gaussian smooth, keep only near-data footprint."""
    m = np.isfinite(layer)
    if not m.any(): return layer, m
    idx = distance_transform_edt(~m, return_distances=False, return_indices=True)
    filled = layer[tuple(idx)]
    sm = gaussian_filter(filled, SMOOTH_SIG)
    foot = binary_dilation(m, iterations=FILL_RAD)
    out = np.where(foot, sm, np.nan)
    return out, foot

# ---- true-color RGB ----
rgb = grid_layers(REFL[:, :, [iR, iG, iB]])
chans = []; foot = None
for k in range(3):
    c, foot = fill_smooth(rgb[k]); chans.append(c)
g3 = np.stack(chans)
cm = np.nanmean(g3.reshape(3, -1), 1); g3 = g3 * (np.nanmean(cm) / (cm[:, None, None] + 1e-9))
lo, hi = np.nanpercentile(g3, [2, 98]); g3 = np.clip((g3 - lo) / (hi - lo + 1e-9), 0, 1) ** 0.8
mean = np.nanmean(g3, 0, keepdims=True); g3 = np.clip(mean + (g3 - mean) * 1.6, 0, 1)
rgb8 = np.where(np.isfinite(g3), g3 * 255, 0).astype(np.uint8)
alpha = (foot & np.isfinite(g3).all(0)).astype(np.uint8) * 255
with rasterio.open(os.path.join(OUT, 'ceva_PASS_truecolor_utm18n.tif'), 'w', driver='GTiff',
                   height=nrow, width=ncol, count=4, dtype='uint8', crs=CRS.from_epsg(32618),
                   transform=tr, photometric='RGB', compress='deflate') as ds:
    for b in range(3): ds.write(rgb8[b], b + 1)
    ds.write(alpha, 4)
    ds.colorinterp = [rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                      rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha]

# ---- NDCI ----
R665 = REFL[:, :, i665]; R705 = REFL[:, :, i705]
ndci = grid_layers(((R705 - R665) / (R705 + R665))[:, :, None])[0]
nd_sm, nd_foot = fill_smooth(ndci)
nd_out = np.where(nd_foot, nd_sm, np.nan).astype('float32')
with rasterio.open(os.path.join(OUT, 'ceva_PASS_ndci_utm18n.tif'), 'w', driver='GTiff',
                   height=nrow, width=ncol, count=1, dtype='float32', crs=CRS.from_epsg(32618),
                   transform=tr, nodata=float('nan'), compress='deflate') as ds:
    ds.write(nd_out, 1)

# ---- previews ----
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ext = [xmin, xmax, ymin, ymax]
plt.figure(figsize=(7, 10)); plt.imshow(np.dstack([rgb8[0], rgb8[1], rgb8[2], alpha]), extent=ext, origin='upper')
plt.title(f'Single-pass TRUE-COLOR — smooth, glint-removed\n{N} frames, UTM18N {RES_M}m')
plt.xlabel('easting'); plt.ylabel('northing'); plt.ticklabel_format(useOffset=False, style='plain')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'ceva_PASS_truecolor_preview.png'), dpi=110)
plt.figure(figsize=(7, 10))
im = plt.imshow(nd_out, extent=ext, origin='upper', cmap='RdYlGn',
                vmin=np.nanpercentile(nd_out, 5), vmax=np.nanpercentile(nd_out, 95))
plt.colorbar(im, label='NDCI'); plt.title('Single-pass NDCI — smooth, glint-removed')
plt.xlabel('easting'); plt.ylabel('northing'); plt.ticklabel_format(useOffset=False, style='plain')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'ceva_PASS_ndci_preview.png'), dpi=110)
print('DONE')
