#!/usr/bin/env python
"""Georeference the existing Spring Silver band mosaic using the now-available GPS log.
Reuses silver_spring2026_bandmosaics.npz (already glint-masked + inter-frame normalized);
just attaches GPS per frame (log time is MM:SS in hour=frame-hour) and rasterizes to UTM.
Outputs georeferenced true-color + NDCI + 3-band chl-a (replaces the frame-space versions).
"""
import os, csv, numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = os.path.expanduser('~/bayspec_results')
RAW = "/Users/MAzarderakhsh/City Tech Dropbox/Marzi Azar/Projects/NWFW/SP2026/SILVER-4/2026-05-12/14-37-23RES1280_EXP0018_GAN0010_BIT16_2EXP0000_GAN0010_RES1280_BIT08/RawImages"
LOGF = open('/tmp/silver_spring_log.txt').read().strip()
STEP, RES_M, SWATH_M = 2, 0.50, 80.0
BANDS = [470, 550, 620, 665, 675, 705, 740, 800]
iR, iG, iB = BANDS.index(620), BANDS.index(550), BANDS.index(470)
i665, i705, i740 = BANDS.index(665), BANDS.index(705), BANDS.index(740)

d = np.load(f'{OUT}/silver_spring2026_bandmosaics.npz')
REFL = np.transpose(d['bands'], (1, 2, 0)).astype(np.float32)   # (N, ns, 8)
N, ns, _ = REFL.shape
print('band mosaic frames:', N)

# reconstruct the same frame list (sorted, every STEP) and their times
names = [n for n in sorted(os.listdir(RAW)) if not n.startswith('.')][::STEP]
assert len(names) == N, f'frame count mismatch {len(names)} vs {N}'
def sod_f(n):
    q = n.split('_')[2].split('-'); h, m, s_, ms = (int(x) for x in q); return h*3600+m*60+s_+ms/1000.0
ft = np.array([sod_f(n) for n in names])
base_hour = int(names[0].split('_')[2].split('-')[0])

# parse log (MM:SS.s in hour=base_hour, or HH:MM:SS AM/PM fallback)
def ptime(t):
    t = t.strip()
    try:
        import datetime; dd = datetime.datetime.strptime(t, '%I:%M:%S.%f %p')
        return dd.hour*3600+dd.minute*60+dd.second+dd.microsecond/1e6
    except Exception: pass
    p = t.split(':')
    if len(p) == 2: return base_hour*3600 + int(p[0])*60 + float(p[1])
    return int(p[0])*3600 + int(p[1])*60 + float(p[2])
gt, glat, glon, gyaw = [], [], [], []
for row in csv.DictReader(open(LOGF, encoding='utf-8-sig')):
    try:
        gt.append(ptime(row['CUSTOM.updateTime [local]'])); glat.append(float(row['OSD.latitude']))
        glon.append(float(row['OSD.longitude'])); gyaw.append(float(row['OSD.yaw [360]']))
    except Exception: pass
gt = np.array(gt); o = np.argsort(gt)
gt, glat, glon = gt[o], np.array(glat)[o], np.array(glon)[o]; gyaw = np.deg2rad(np.array(gyaw)[o])
print(f'GPS {len(gt)} rows span {gt.min():.0f}-{gt.max():.0f}s ; frames {ft.min():.0f}-{ft.max():.0f}s')

flat = np.interp(ft, gt, glat); flon = np.interp(ft, gt, glon)
fyaw = np.arctan2(np.interp(ft, gt, np.sin(gyaw)), np.interp(ft, gt, np.cos(gyaw)))
zone = int((flon.mean()+180)//6)+1; EPSG = (32600 if flat.mean() >= 0 else 32700)+zone
cE, cN = warp_transform(CRS.from_epsg(4326), CRS.from_epsg(EPSG), list(flon), list(flat))
cE, cN = np.array(cE), np.array(cN)
off = (np.arange(ns)-(ns-1)/2)*(SWATH_M/ns); bear = fyaw[:, None]+np.pi/2
sE = cE[:, None]+off[None, :]*np.sin(bear); sN = cN[:, None]+off[None, :]*np.cos(bear)
valid = np.isfinite(REFL).any(2) & np.isfinite(sE) & np.isfinite(sN)
xmin, xmax = sE[valid].min()-1, sE[valid].max()+1; ymin, ymax = sN[valid].min()-1, sN[valid].max()+1
ncol = int((xmax-xmin)/RES_M); nrow = int((ymax-ymin)/RES_M)
col = np.clip(((sE-xmin)/RES_M).astype(int), 0, ncol-1); rowi = np.clip(((ymax-sN)/RES_M).astype(int), 0, nrow-1)
tr = from_origin(xmin, ymax, RES_M, RES_M)
print(f'EPSG {EPSG} grid {nrow}x{ncol}')

def grid_layers(v):
    L = v.shape[2]; out = np.zeros((L, nrow, ncol)); cnt = np.zeros((L, nrow, ncol))
    rr = np.broadcast_to(rowi[:, :, None], v.shape); cc = np.broadcast_to(col[:, :, None], v.shape); ok = np.isfinite(v)
    for l in range(L):
        m = ok[:, :, l]; np.add.at(out[l], (rr[:, :, l][m], cc[:, :, l][m]), v[:, :, l][m]); np.add.at(cnt[l], (rr[:, :, l][m], cc[:, :, l][m]), 1)
    with np.errstate(invalid='ignore'): return np.where(cnt > 0, out/cnt, np.nan)
def fill_smooth(a):
    m = np.isfinite(a); idx = distance_transform_edt(~m, return_distances=False, return_indices=True)
    foot = binary_dilation(m, iterations=5); return np.where(foot, gaussian_filter(a[tuple(idx)], 1.5), np.nan), foot
def wtif(path, arr, **kw):
    with rasterio.open(path, 'w', driver='GTiff', height=nrow, width=ncol, count=arr.shape[0], dtype='float32',
                       crs=CRS.from_epsg(EPSG), transform=tr, compress='deflate', nodata=float('nan'), **kw) as ds:
        for b in range(arr.shape[0]): ds.write(arr[b].astype('float32'), b+1)

bands_g = grid_layers(REFL)
wtif(f'{OUT}/silver_spring2026_bands_utm.tif', bands_g)
R665, R705, R740 = bands_g[i665], bands_g[i705], bands_g[i740]
with np.errstate(divide='ignore', invalid='ignore'):
    ndci = (R705-R665)/(R705+R665); chl3 = (1.0/R665 - 1.0/R705)*R740
nd, ndf = fill_smooth(ndci); c3, c3f = fill_smooth(chl3)
wtif(f'{OUT}/silver_spring2026_ndci_utm.tif', np.where(ndf, nd, np.nan)[None])
wtif(f'{OUT}/silver_spring2026_chl3band_utm.tif', np.where(c3f, c3, np.nan)[None])
# true color
rgb = grid_layers(REFL[:, :, [iR, iG, iB]]); ch = []; foot = None
for k in range(3): c, foot = fill_smooth(rgb[k]); ch.append(c)
g3 = np.stack(ch); cm = np.nanmean(g3.reshape(3, -1), 1); g3 = g3*(np.nanmean(cm)/(cm[:, None, None]+1e-9))
lo, hi = np.nanpercentile(g3, [2, 98]); g3 = np.clip((g3-lo)/(hi-lo+1e-9), 0, 1)**0.8
mean = np.nanmean(g3, 0, keepdims=True); g3 = np.clip(mean+(g3-mean)*1.5, 0, 1)
rgb8 = np.where(np.isfinite(g3), g3*255, 0).astype(np.uint8); alpha = foot.astype(np.uint8)*255
with rasterio.open(f'{OUT}/silver_spring2026_truecolor_utm.tif', 'w', driver='GTiff', height=nrow, width=ncol,
                   count=4, dtype='uint8', crs=CRS.from_epsg(EPSG), transform=tr, photometric='RGB', compress='deflate') as ds:
    for b in range(3): ds.write(rgb8[b], b+1)
    ds.write(alpha, 4)
ext = [xmin, xmax, ymin, ymax]
fig, ax = plt.subplots(1, 3, figsize=(20, 7))
ax[0].imshow(np.dstack([rgb8[0], rgb8[1], rgb8[2], alpha]), extent=ext, origin='upper'); ax[0].set_title('Spring true-color (georef)')
for a, img, t in [(ax[1], np.where(ndf, nd, np.nan), 'NDCI'), (ax[2], np.where(c3f, c3, np.nan), '3-band chl-a')]:
    v = img[np.isfinite(img)]; im = a.imshow(img, extent=ext, origin='upper', cmap='RdYlGn', vmin=np.percentile(v, 5), vmax=np.percentile(v, 95))
    a.set_title(t); plt.colorbar(im, ax=a, fraction=0.04)
for a in ax: a.ticklabel_format(useOffset=False, style='plain')
plt.suptitle(f'SILVER SPRING 2026 — georeferenced (EPSG {EPSG})'); plt.tight_layout()
plt.savefig(f'{OUT}/silver_spring2026_georef_preview.png', dpi=110)
# remove stale frame-space pngs
for f in ('silver_spring2026_truecolor_framespace.png', 'silver_spring2026_ndci_framespace.png'):
    p = f'{OUT}/{f}'
    if os.path.exists(p): os.remove(p)
print('DONE: wrote georeferenced Spring true-color + NDCI + 3-band; NDCI median %.4f chl3 median %.4f' %
      (np.nanmedian(nd[np.isfinite(nd)]), np.nanmedian(c3[np.isfinite(c3)])))
