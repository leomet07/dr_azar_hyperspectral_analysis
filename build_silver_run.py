#!/usr/bin/env python
"""Generalized Silver Lake run: calibrate + index + mosaic for one flight.

Usage: build_silver_run.py <capture_dir> <flight_log_csv|NONE> <out_prefix>

- auto-detects bit depth (8 vs 16) from raw file size
- white-saturation check -> SAT_BIAS (1.0 if clean)
- georeferenced mosaics (true-color + NDCI + per-band) in auto UTM when a GPS log is given
- frame-space mosaics when no GPS log
Outputs -> ~/bayspec_results/<out_prefix>_*
"""
import os, sys, re, csv, glob, math, datetime, time
import numpy as np
import xml.etree.ElementTree as ET
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

CAP, LOGF, PREFIX = sys.argv[1], sys.argv[2], sys.argv[3]
STEP, RES_M = 2, 0.50
BANDS_NM = [470, 550, 620, 665, 675, 705, 740, 800]
RGB_NM = (620, 550, 470)
GLINT_PCTL, SMOOTH_SIG, FILL_RAD = 60, 1.5, 5
OUT = os.path.expanduser('~/bayspec_results')
LOG = os.path.join(OUT, f'{PREFIX}_progress.log')
os.makedirs(OUT, exist_ok=True); open(LOG, 'w').close()
def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {PREFIX}: {m}'; print(line, flush=True)
    open(LOG, 'a').write(line + '\n')

# ---- sensor cfg ----
xmlp = glob.glob(os.path.join(CAP, '*.xml'))[0]
s = ET.parse(xmlp).getroot().find('sensor'); gv = lambda t: float(s.find(t).text)
W, H, LPB, BPX = int(gv('width')), int(gv('height')), int(gv('linesperband')), int(gv('bandpixels'))
nb, ns = H // LPB, W // BPX
by = np.arange(nb) * LPB + LPB / 2
wls = gv('c0') + gv('c1') * by + gv('c2') * by ** 2
bidx = [int(np.argmin(np.abs(wls - x))) for x in BANDS_NM]
def b_of(nm): return BANDS_NM.index(min(BANDS_NM, key=lambda x: abs(x - nm)))
iR, iG, iB = b_of(620), b_of(550), b_of(470); i665, i705, i800 = b_of(665), b_of(705), b_of(800)
log(f'sensor {W}x{H} -> {nb}x{ns}; lambda {wls[0]:.0f}-{wls[-1]:.0f}nm')

def load_frame(p):
    npix = W * H; nbytes = os.path.getsize(p)
    dt = np.uint8 if nbytes == npix else np.uint16
    return np.fromfile(p, dtype=dt).astype(np.float32).reshape(H, W)
def cube(p):
    f = load_frame(p); return f[:nb*LPB, :ns*BPX].reshape(nb, LPB, ns, BPX).mean(axis=(1, 3))
def avg_ref(folder):
    fs = [x for x in sorted(os.listdir(folder)) if not x.startswith('.') and not x.lower().endswith(('.xml',))][:12]
    return np.mean([cube(os.path.join(folder, f)) for f in fs], axis=0)

dark_full = avg_ref(os.path.join(CAP, 'Dark'))
white_full = avg_ref(os.path.join(CAP, 'WhiteRef'))
wmax = white_full.max(); cap = next(c for c in (255, 1023, 4095, 65535) if wmax <= c)
satf = 100 * (white_full >= cap).mean()
SAT_BIAS = 1.0 if satf < 1 else 3.5
log(f'white max {wmax:.0f} cap {cap} sat {satf:.2f}% -> SAT_BIAS={SAT_BIAS}')
dark, white = dark_full[bidx], white_full[bidx]
denom = np.where(np.abs(white - dark) < 0.5, np.nan, white - dark)
panel = white_full.mean(0) > white_full.mean() * 0.4

# ---- frame list (+ GPS window if log) ----
RAW = os.path.join(CAP, 'RawImages')
def sod_f(n):
    q = n.split('_')[2].split('-'); h, m, s_, ms = (int(x) for x in q); return h*3600+m*60+s_+ms/1000.0
HAVE_GPS = LOGF.upper() != 'NONE' and os.path.exists(LOGF)
gt = glat = glon = gyaw = None
if HAVE_GPS:
    def sod_a(t):
        d = datetime.datetime.strptime(t.strip(), '%I:%M:%S.%f %p'); return d.hour*3600+d.minute*60+d.second+d.microsecond/1e6
    gt, glat, glon, gyaw = [], [], [], []
    for row in csv.DictReader(open(LOGF, encoding='utf-8-sig')):
        try:
            gt.append(sod_a(row['CUSTOM.updateTime [local]'])); glat.append(float(row['OSD.latitude']))
            glon.append(float(row['OSD.longitude'])); gyaw.append(float(row['OSD.yaw [360]']))
        except Exception: pass
    gt = np.array(gt); o = np.argsort(gt)
    gt, glat, glon = gt[o], np.array(glat)[o], np.array(glon)[o]; gyaw = np.deg2rad(np.array(gyaw)[o])
    log(f'GPS rows={len(gt)} span {gt.min():.0f}-{gt.max():.0f}s')

names = []
for n in sorted(os.listdir(RAW)):
    if n.startswith('.'): continue
    try: t = sod_f(n)
    except Exception: continue
    if (not HAVE_GPS) or (gt.min() <= t <= gt.max()): names.append(n)
names = names[::STEP]
ft = np.array([sod_f(n) for n in names])
log(f'{len(names)} frames (every {STEP}){" in GPS window" if HAVE_GPS else " (no GPS)"}')

# ---- read + calibrate ----
B = len(BANDS_NM); REFL = np.full((len(names), ns, B), np.nan, np.float32)
t0 = time.time()
for i, n in enumerate(names):
    try: cu = cube(os.path.join(RAW, n))
    except Exception: continue
    R = (cu[bidx] - dark) / denom * 0.99 / SAT_BIAS; R[:, ~panel] = np.nan
    REFL[i] = R.T
    if i % 3000 == 0: log(f'  read {i}/{len(names)} {time.time()-t0:.0f}s')
log(f'calibrated {time.time()-t0:.0f}s; median R {np.nanmedian(REFL):.4f}')

# ---- glint mask + inter-frame normalization ----
nir = REFL[:, :, i800]; fin = np.isfinite(nir)
nthr = np.nanpercentile(nir[fin], GLINT_PCTL)
vis = np.nanmean(REFL[:, :, [iR, iG, iB]], axis=2)
med = np.nanmedian(vis[fin]); mad = np.nanmedian(np.abs(vis[fin] - med))
glint = (~fin) | (nir > nthr) | (vis > med + 3*1.4826*mad)
REFL[np.broadcast_to(glint[:, :, None], REFL.shape)] = np.nan
fb = np.nanmedian(REFL.reshape(len(names), -1), axis=1)
REFL = REFL * np.where(fb > 1e-6, np.nanmedian(fb)/fb, np.nan)[:, None, None]
log(f'glint removed {100*glint.mean():.1f}%; inter-frame normalized')

R665 = REFL[:, :, i665]; R705 = REFL[:, :, i705]
NDCI = (R705 - R665) / (R705 + R665)

def truecolor_from(rgb3):
    g = rgb3.copy(); cm = np.nanmean(g.reshape(-1, 3), 0); g = g * (np.nanmean(cm)/(cm+1e-9))
    g = g.reshape(rgb3.shape); lo, hi = np.nanpercentile(g, [2, 98])
    g = np.clip((g-lo)/(hi-lo+1e-9), 0, 1) ** 0.8
    mean = np.nanmean(g, axis=-1, keepdims=True); return np.clip(mean + (g-mean)*1.5, 0, 1)

if HAVE_GPS:
    flat = np.interp(ft, gt, glat); flon = np.interp(ft, gt, glon)
    fyaw = np.arctan2(np.interp(ft, gt, np.sin(gyaw)), np.interp(ft, gt, np.cos(gyaw)))
    zone = int((flon.mean()+180)//6)+1; EPSG = (32600 if flat.mean() >= 0 else 32700)+zone
    cE, cN = warp_transform(CRS.from_epsg(4326), CRS.from_epsg(EPSG), list(flon), list(flat))
    cE, cN = np.array(cE), np.array(cN)
    SWATH_M = 80.0; off = (np.arange(ns)-(ns-1)/2)*(SWATH_M/ns); bear = fyaw[:, None]+np.pi/2
    sE = cE[:, None]+off[None, :]*np.sin(bear); sN = cN[:, None]+off[None, :]*np.cos(bear)
    valid = np.isfinite(REFL).any(2) & np.isfinite(sE) & np.isfinite(sN)
    xmin, xmax = sE[valid].min()-1, sE[valid].max()+1; ymin, ymax = sN[valid].min()-1, sN[valid].max()+1
    ncol = int((xmax-xmin)/RES_M); nrow = int((ymax-ymin)/RES_M)
    col = np.clip(((sE-xmin)/RES_M).astype(int), 0, ncol-1); rowi = np.clip(((ymax-sN)/RES_M).astype(int), 0, nrow-1)
    tr = from_origin(xmin, ymax, RES_M, RES_M)
    log(f'EPSG {EPSG}; grid {nrow}x{ncol}')
    def grid_layers(v):
        L = v.shape[2]; out = np.zeros((L, nrow, ncol)); cnt = np.zeros((L, nrow, ncol))
        rr = np.broadcast_to(rowi[:, :, None], v.shape); cc = np.broadcast_to(col[:, :, None], v.shape); ok = np.isfinite(v)
        for l in range(L):
            m = ok[:, :, l]; np.add.at(out[l], (rr[:, :, l][m], cc[:, :, l][m]), v[:, :, l][m]); np.add.at(cnt[l], (rr[:, :, l][m], cc[:, :, l][m]), 1)
        with np.errstate(invalid='ignore'): return np.where(cnt > 0, out/cnt, np.nan)
    def fill_smooth(layer):
        m = np.isfinite(layer); idx = distance_transform_edt(~m, return_distances=False, return_indices=True)
        return np.where(binary_dilation(m, iterations=FILL_RAD), gaussian_filter(layer[tuple(idx)], SMOOTH_SIG), np.nan), binary_dilation(m, iterations=FILL_RAD)
    def wtif(path, arr, dtype, **kw):
        with rasterio.open(path, 'w', driver='GTiff', height=nrow, width=ncol, count=arr.shape[0], dtype=dtype,
                           crs=CRS.from_epsg(EPSG), transform=tr, compress='deflate', **kw) as ds:
            for b in range(arr.shape[0]): ds.write(arr[b], b+1)
    bands_g = grid_layers(REFL)
    wtif(f'{OUT}/{PREFIX}_bands_utm.tif', bands_g.astype('float32'), 'float32', nodata=float('nan'))
    ndg, ndf = fill_smooth(grid_layers(NDCI[:, :, None])[0])
    wtif(f'{OUT}/{PREFIX}_ndci_utm.tif', np.where(ndf, ndg, np.nan)[None].astype('float32'), 'float32', nodata=float('nan'))
    rgb = grid_layers(REFL[:, :, [iR, iG, iB]]); ch = []; foot = None
    for k in range(3): c, foot = fill_smooth(rgb[k]); ch.append(c)
    tc = truecolor_from(np.stack(ch, -1)); rgb8 = np.where(np.isfinite(tc), tc*255, 0).astype(np.uint8).transpose(2, 0, 1)
    alpha = (foot).astype(np.uint8)*255
    with rasterio.open(f'{OUT}/{PREFIX}_truecolor_utm.tif', 'w', driver='GTiff', height=nrow, width=ncol, count=4,
                       dtype='uint8', crs=CRS.from_epsg(EPSG), transform=tr, photometric='RGB', compress='deflate') as ds:
        for b in range(3): ds.write(rgb8[b], b+1)
        ds.write(alpha, 4)
    ext = [xmin, xmax, ymin, ymax]
    plt.figure(figsize=(9, 9)); plt.imshow(np.dstack([rgb8[0], rgb8[1], rgb8[2], alpha]), extent=ext, origin='upper')
    plt.title(f'{PREFIX} true-color (georef, glint-free)'); plt.ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout(); plt.savefig(f'{OUT}/{PREFIX}_truecolor_preview.png', dpi=110); plt.close()
    nd = np.where(ndf, ndg, np.nan)
    plt.figure(figsize=(9, 9)); im = plt.imshow(nd, extent=ext, origin='upper', cmap='RdYlGn',
              vmin=np.nanpercentile(nd, 5), vmax=np.nanpercentile(nd, 95)); plt.colorbar(im, label='NDCI')
    plt.title(f'{PREFIX} NDCI (georef)'); plt.ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout(); plt.savefig(f'{OUT}/{PREFIX}_ndci_preview.png', dpi=110); plt.close()
    log(f'wrote georeferenced GeoTIFFs + previews (EPSG {EPSG})')
else:
    # frame-space mosaics (no GPS): stack frames (along-track) x across-track
    def smooth2d(a):
        m = np.isfinite(a)
        if not m.any(): return a
        idx = distance_transform_edt(~m, return_distances=False, return_indices=True)
        return np.where(binary_dilation(m, iterations=FILL_RAD), gaussian_filter(a[tuple(idx)], SMOOTH_SIG), np.nan)
    rgb = truecolor_from(REFL[:, :, [iR, iG, iB]])
    rgb = np.stack([smooth2d(rgb[:, :, k]) for k in range(3)], -1)
    plt.figure(figsize=(7, 11)); plt.imshow(np.clip(np.nan_to_num(rgb), 0, 1), aspect='auto', origin='lower')
    plt.title(f'{PREFIX} true-color FRAME-SPACE (no GPS)\nY=along-track frame, X=across-track')
    plt.xlabel('across-track px'); plt.ylabel('frame'); plt.tight_layout()
    plt.savefig(f'{OUT}/{PREFIX}_truecolor_framespace.png', dpi=110); plt.close()
    nd = smooth2d(NDCI)
    plt.figure(figsize=(7, 11)); im = plt.imshow(nd, aspect='auto', origin='lower', cmap='RdYlGn',
              vmin=np.nanpercentile(nd, 5), vmax=np.nanpercentile(nd, 95)); plt.colorbar(im, label='NDCI')
    plt.title(f'{PREFIX} NDCI FRAME-SPACE (no GPS)'); plt.xlabel('across-track px'); plt.ylabel('frame')
    plt.tight_layout(); plt.savefig(f'{OUT}/{PREFIX}_ndci_framespace.png', dpi=110); plt.close()
    np.savez_compressed(f'{OUT}/{PREFIX}_bandmosaics.npz',
                        bands=np.stack([REFL[:, :, b] for b in range(B)]), band_nm=np.array(BANDS_NM),
                        ndci=NDCI, sat_bias=SAT_BIAS)
    log('wrote frame-space mosaics + bandmosaics.npz (georeferencing pending a GPS log)')
log('DONE')
