#!/usr/bin/env python
"""Full-survey lake processing -> georeferenced band + index + true-color mosaics.

Pipeline (addresses frame-to-frame banding, glint, and gray-vs-blue/green color):
  1. read key bands for every GPS-covered frame (whole flight)
  2. CALIBRATE to reflectance per band/column: (DN-dark)/(white-dark)
       -> this removes the across-track vignette (white encodes it per column)
  3. SUN-GLINT / land mask: water is dark in NIR; drop pixels with high NIR reflectance
  4. INTER-FRAME NORMALIZATION: each frame gets ONE scalar gain so all frames share a
       common brightness reference -> removes the along-track striping / "different
       reference per frame" look, without flattening spectral (color) differences
  5. indices: NDCI, red-edge (R705), chl-a ratio (R705/R665)
  6. georeference every scan line via the DJI GPS log -> rasterize to UTM 18N
  7. write multiband (per-band) GeoTIFF, index GeoTIFFs, and a true-color RGB GeoTIFF + PNGs
"""
import numpy as np, csv, json, datetime, os, time
import xml.etree.ElementTree as ET
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

STEP        = 2
SWATH_M     = 80.0
RES_M       = 0.50
BANDS_NM    = [470, 550, 620, 665, 675, 705, 740, 800]   # saved per-band; RGB+indices drawn from these
RGB_NM      = (620, 550, 470)
GLINT_NIR   = 800            # band used to detect glint/land
SAT_BIAS    = 3.5
OUT         = os.path.expanduser('~/bayspec_results')
LOGTXT      = os.path.join(OUT, 'full_mosaic_progress.log')

def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'; print(line, flush=True)
    with open(LOGTXT, 'a') as f: f.write(line + '\n')

sel = json.load(open('/tmp/v4_inputs.json'))
XML, DARK, WHITE = sel['XML'], sel['DARK'], sel['WHITE']
RAW = os.path.dirname(sel['LAKES'][0])
LOGF = open('/tmp/flightlog_path.txt').read().strip()
os.makedirs(OUT, exist_ok=True); open(LOGTXT, 'w').close()

# ---- sensor cfg ----
r = ET.parse(XML).getroot(); s = r.find('sensor'); g = lambda t, f=float: f(s.find(t).text)
W, H, LPB, BPX = int(g('width', int)), int(g('height', int)), int(g('linesperband', int)), int(g('bandpixels', int))
nb, ns = H // LPB, W // BPX
by = np.arange(nb) * LPB + LPB / 2
wls = g('c0') + g('c1') * by + g('c2') * by ** 2
bidx = [int(np.argmin(np.abs(wls - x))) for x in BANDS_NM]
def bi(nm): return BANDS_NM.index(min(BANDS_NM, key=lambda x: abs(x - nm)))  # index into BANDS_NM
iR, iG, iB = bi(RGB_NM[0]), bi(RGB_NM[1]), bi(RGB_NM[2])
i470, i550, i620, i665, i705, i800 = bi(470), bi(550), bi(620), bi(665), bi(705), bi(800)

def full_cube(path):
    a = np.fromfile(path, dtype=np.uint8)
    if a.size != W * H: return None
    a = a.reshape(H, W).astype(np.float32)
    return a[:nb * LPB, :ns * BPX].reshape(nb, LPB, ns, BPX).mean(axis=(1, 3))   # (nb, ns)

dark_full = full_cube(DARK); white_full = full_cube(WHITE)
dark = dark_full[bidx, :]                       # (B, ns)
white = white_full[bidx, :]
denom = white - dark
denom = np.where(np.abs(denom) < 0.5, np.nan, denom)
panel_cols = white_full.mean(axis=0) > 100      # on-panel across-track columns

# ---- flight log ----
def sod_ampm(t):
    d = datetime.datetime.strptime(t.strip(), '%I:%M:%S.%f %p')
    return d.hour*3600 + d.minute*60 + d.second + d.microsecond/1e6
gt, glat, glon, gyaw = [], [], [], []
with open(LOGF, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        try:
            gt.append(sod_ampm(row['CUSTOM.updateTime [local]']))
            glat.append(float(row['OSD.latitude'])); glon.append(float(row['OSD.longitude']))
            gyaw.append(float(row['OSD.yaw [360]']))
        except Exception: pass
gt = np.array(gt); glat = np.array(glat); glon = np.array(glon); gyaw = np.deg2rad(np.array(gyaw))
o = np.argsort(gt); gt, glat, glon, gyaw = gt[o], glat[o], glon[o], gyaw[o]

def sod_fname(n):
    p = n.split('_')[2].split('-'); h, m, s_, ms = (int(x) for x in p)
    return h*3600 + m*60 + s_ + ms/1000.0
allf = sorted(os.listdir(RAW))
frames = []
for n in allf:
    try: t = sod_fname(n)
    except Exception: continue
    if gt.min() <= t <= gt.max(): frames.append((n, t))
frames = frames[::STEP]
names = [n for n, _ in frames]; ft = np.array([t for _, t in frames])
log(f'GPS rows={len(gt)} | frames in window (every {STEP})={len(names)}')

# ---- read selected bands, calibrate to reflectance ----
B = len(BANDS_NM)
REFL = np.full((len(names), ns, B), np.nan, np.float32)
t0 = time.time()
for i, n in enumerate(names):
    cu = full_cube(os.path.join(RAW, n))
    if cu is None: continue
    sub = cu[bidx, :]                                  # (B, ns)
    R = (sub - dark) / denom * 0.99 / SAT_BIAS         # reflectance (vignette removed via /white)
    R[:, ~panel_cols] = np.nan
    REFL[i] = R.T                                      # (ns, B)
    if i % 2000 == 0:
        el = time.time() - t0
        log(f'  read {i}/{len(names)} elapsed {el:.0f}s eta {el/(i+1)*(len(names)-i-1):.0f}s')
log(f'read+calibrate done {time.time()-t0:.0f}s')

# ---- sun-glint / land mask: water is dark in NIR ----
nir = REFL[:, :, i800]
finite = np.isfinite(nir)
thr = np.nanpercentile(nir[finite], 70)               # keep the darker-NIR (water) pixels
thr = max(thr, 0.0)
water = finite & (nir <= thr)
log(f'glint/land mask: NIR<= {thr:.4f} keeps {100*water.mean():.1f}% of finite px as water')
MASK = water[:, :, None]
REFLm = np.where(np.broadcast_to(MASK, REFL.shape), REFL, np.nan)

# ---- INTER-FRAME NORMALIZATION: one gain per frame so all frames share a reference ----
# brightness of each frame = median reflectance over its valid water pixels, all bands
fb = np.nanmedian(REFLm.reshape(len(names), -1), axis=1)        # per-frame brightness
glob = np.nanmedian(fb)
gain = np.where(np.isfinite(fb) & (fb > 1e-6), glob / fb, np.nan)
REFLn = REFLm * gain[:, None, None]
log(f'inter-frame normalization applied (global brightness {glob:.4f})')

# ---- indices ----
R665 = REFLn[:, :, i665]; R705 = REFLn[:, :, i705]
NDCI = (R705 - R665) / (R705 + R665)
REDEDGE = R705
CHL = R705 / R665

# ---- georeference ----
flat_lat = np.interp(ft, gt, glat); flat_lon = np.interp(ft, gt, glon)
fyaw = np.arctan2(np.interp(ft, gt, np.sin(gyaw)), np.interp(ft, gt, np.cos(gyaw)))
cE, cN = warp_transform(CRS.from_epsg(4326), CRS.from_epsg(32618), list(flat_lon), list(flat_lat))
cE = np.array(cE); cN = np.array(cN)
j = np.arange(ns); off = (j - (ns - 1) / 2.0) * (SWATH_M / ns)
bear = fyaw[:, None] + np.pi / 2
sE = cE[:, None] + off[None, :] * np.sin(bear)
sN = cN[:, None] + off[None, :] * np.cos(bear)

# bounds from valid water samples (tight around the lake)
anyvalid = np.isfinite(REFLn).any(axis=2) & np.isfinite(sE) & np.isfinite(sN)
xmin, xmax = sE[anyvalid].min() - 1, sE[anyvalid].max() + 1
ymin, ymax = sN[anyvalid].min() - 1, sN[anyvalid].max() + 1
ncol = int(np.ceil((xmax - xmin) / RES_M)); nrow = int(np.ceil((ymax - ymin) / RES_M))
col = np.clip(((sE - xmin) / RES_M).astype(int), 0, ncol - 1)
rowi = np.clip(((ymax - sN) / RES_M).astype(int), 0, nrow - 1)
log(f'grid {nrow}x{ncol} @ {RES_M}m  bbox E{xmin:.0f}-{xmax:.0f} N{ymin:.0f}-{ymax:.0f}')

def grid_layers(values):                          # values (N, ns, L) -> (L, nrow, ncol) mean
    L = values.shape[2]
    out = np.zeros((L, nrow, ncol)); cnt = np.zeros((L, nrow, ncol))
    rr = np.broadcast_to(rowi[:, :, None], values.shape)
    cc = np.broadcast_to(col[:, :, None], values.shape)
    ok = np.isfinite(values)
    for l in range(L):
        m = ok[:, :, l]
        np.add.at(out[l], (rr[:, :, l][m], cc[:, :, l][m]), values[:, :, l][m])
        np.add.at(cnt[l], (rr[:, :, l][m], cc[:, :, l][m]), 1)
    with np.errstate(invalid='ignore'):
        return np.where(cnt > 0, out / cnt, np.nan), cnt

tr = from_origin(xmin, ymax, RES_M, RES_M)
def write_tif(path, arr, dtype, nodata=None, photometric=None, alpha=None, descr=None):
    count = arr.shape[0] + (1 if alpha is not None else 0)
    kw = dict(driver='GTiff', height=nrow, width=ncol, count=count, dtype=dtype,
              crs=CRS.from_epsg(32618), transform=tr, compress='deflate')
    if nodata is not None: kw['nodata'] = nodata
    if photometric: kw['photometric'] = photometric
    with rasterio.open(path, 'w', **kw) as ds:
        for b in range(arr.shape[0]): ds.write(arr[b], b + 1)
        if alpha is not None: ds.write(alpha, arr.shape[0] + 1)
        if descr:
            for b, d in enumerate(descr): ds.set_band_description(b + 1, d)

# ---- per-band reflectance mosaic (multiband) ----
band_grid, _ = grid_layers(REFLn)
write_tif(os.path.join(OUT, 'ceva_FULL_bands_utm18n.tif'),
          band_grid.astype('float32'), 'float32', nodata=float('nan'),
          descr=[f'{nm}nm' for nm in BANDS_NM])
log('wrote per-band mosaic')

# ---- index mosaics ----
idx = np.stack([NDCI, REDEDGE, CHL], axis=2)
idx_grid, _ = grid_layers(idx)
for k, nm in enumerate(['ndci', 'rededge_R705', 'chl_R705_R665']):
    write_tif(os.path.join(OUT, f'ceva_FULL_{nm}_utm18n.tif'),
              idx_grid[k:k+1].astype('float32'), 'float32', nodata=float('nan'))
log('wrote index mosaics (NDCI, red-edge, chl ratio)')

# ---- true-color RGB (white-balanced + saturation boost) ----
rgb = REFLn[:, :, [iR, iG, iB]]
rgb_grid, cnt = grid_layers(rgb)                  # (3, nrow, ncol)
g3 = rgb_grid.copy()
cm = np.nanmean(g3.reshape(3, -1), axis=1)
g3 = g3 * (np.nanmean(cm) / (cm[:, None, None] + 1e-9))     # gray-world WB
lo, hi = np.nanpercentile(g3, [2, 98]); g3 = np.clip((g3 - lo) / (hi - lo + 1e-9), 0, 1) ** 0.8
# saturation boost so blue/green water reads as color, not gray
mean = np.nanmean(g3, axis=0, keepdims=True)
g3 = np.clip(mean + (g3 - mean) * 1.6, 0, 1)
rgb8 = np.where(np.isfinite(g3), g3 * 255, 0).astype(np.uint8)
alpha = (np.isfinite(rgb_grid).all(axis=0)).astype(np.uint8) * 255
write_tif(os.path.join(OUT, 'ceva_FULL_truecolor_utm18n.tif'),
          rgb8, 'uint8', photometric='RGB', alpha=alpha)
log('wrote true-color RGB mosaic')

# ---- PNG previews ----
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ext = [xmin, xmax, ymin, ymax]
disp = np.dstack([rgb8[0], rgb8[1], rgb8[2], alpha])
plt.figure(figsize=(9, 9)); plt.imshow(disp, extent=ext, origin='upper')
plt.title(f'Full-survey TRUE-COLOR (glint-masked, inter-frame normalized)\n{len(names)} frames, UTM18N {RES_M}m')
plt.xlabel('easting'); plt.ylabel('northing'); plt.ticklabel_format(useOffset=False, style='plain')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'ceva_FULL_truecolor_preview.png'), dpi=110)

plt.figure(figsize=(9, 9))
nd = idx_grid[0]
im = plt.imshow(nd, extent=ext, origin='upper', cmap='RdYlGn', vmin=np.nanpercentile(nd,5), vmax=np.nanpercentile(nd,95))
plt.colorbar(im, label='NDCI (chl-a)'); plt.title('Full-survey NDCI mosaic')
plt.xlabel('easting'); plt.ylabel('northing'); plt.ticklabel_format(useOffset=False, style='plain')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'ceva_FULL_ndci_preview.png'), dpi=110)

# montage of per-band mosaics
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
for ax, b in zip(axes.ravel(), range(B)):
    bg = band_grid[b]
    ax.imshow(bg, extent=ext, origin='upper', cmap='viridis',
              vmin=np.nanpercentile(bg,5), vmax=np.nanpercentile(bg,95))
    ax.set_title(f'{BANDS_NM[b]} nm'); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('Per-band reflectance mosaics'); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'ceva_FULL_bands_montage.png'), dpi=100)
log('wrote PNG previews'); log('DONE')
