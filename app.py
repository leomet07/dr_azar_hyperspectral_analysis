#!/usr/bin/env python
"""BaySpec OCI-F single-frame reflectance viewer/exporter.

Pick an OCI-F XML calibration file, a dark reference frame, a white reference
frame, a RawImages folder, and a GPS flight-log CSV. Scrub to a time, see the
matching raw frame, calibrate it to reflectance, and export it as a CSV
(rows = wavelength, columns = spatial pixel).

Calibration logic ported from bayspec-silver-sp26.ipynb / build_silver_run.py.
"""
import csv
import datetime
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

PANEL_R = 0.99


def parse_ocif_xml(path):
    s = ET.parse(path).getroot().find('sensor')
    g = lambda t: s.find(t).text
    cfg = dict(width=int(g('width')), height=int(g('height')),
               linesperband=int(g('linesperband')), bandpixels=int(g('bandpixels')),
               c0=float(g('c0')), c1=float(g('c1')), c2=float(g('c2')))
    cfg['n_bands'] = cfg['height'] // cfg['linesperband']
    cfg['n_spatial'] = cfg['width'] // cfg['bandpixels']
    by = np.arange(cfg['n_bands']) * cfg['linesperband'] + cfg['linesperband'] / 2
    cfg['wavelengths'] = cfg['c0'] + cfg['c1'] * by + cfg['c2'] * by ** 2
    return cfg


def load_raw_frame(path, cfg):
    npix = cfg['width'] * cfg['height']
    nbytes = os.path.getsize(path)
    if nbytes == npix:
        dt = np.uint8
    elif nbytes == npix * 2:
        dt = np.uint16
    else:
        raise ValueError(f'{path}: {nbytes} bytes != {npix} (8-bit) or {npix * 2} (16-bit)')
    frame = np.fromfile(path, dtype=dt).astype(np.float32).reshape(cfg['height'], cfg['width'])
    return frame, (8 if dt == np.uint8 else 16)


def bin_to_cube(frame, cfg):
    LPB, BPX, nb, ns = cfg['linesperband'], cfg['bandpixels'], cfg['n_bands'], cfg['n_spatial']
    f = frame[:nb * LPB, :ns * BPX]
    return f.reshape(nb, LPB, ns, BPX).mean(axis=(1, 3))


def sod_from_frame_filename(name):
    q = Path(name).name.split('_')[2].split('-')
    h, m, s, ms = (int(x) for x in q)
    return h * 3600 + m * 60 + s + ms / 1000.0


def sod_ampm(t):
    t = t.strip()
    parts = t.split(':')
    if len(parts) == 2:  # DJI MM:SS.f format (no hour)
        return int(parts[0]) * 60 + float(parts[1])
    d = datetime.datetime.strptime(t, '%I:%M:%S.%f %p')
    return d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6


def fmt_hms(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('BaySpec OCI-F — Single Frame Reflectance')
        self.geometry('1020x820')

        self.paths = {
            'xml': tk.StringVar(), 'dark': tk.StringVar(), 'white': tk.StringVar(),
            'rawdir': tk.StringVar(), 'flightlog': tk.StringVar(),
        }
        self.gain_white = tk.StringVar(value='30')
        self.gain_lake = tk.StringVar(value='10')
        self.time_var = tk.DoubleVar()

        self.cfg = None
        self.dark_cube = None
        self.denom = None
        self.panel = None
        self.sat_bias = None
        self.frame_times = None
        self.frame_paths = None
        self.matched_idx = None
        self.current_R = None
        self.current_wls = None
        self.current_frame_name = None

        self._build_widgets()

    def _build_widgets(self):
        pick_frame = ttk.LabelFrame(self, text='Inputs')
        pick_frame.pack(fill='x', padx=8, pady=6)

        rows = [
            ('xml', 'OCI-F XML calibration file', [('XML', '*.xml'), ('All files', '*.*')], False),
            ('dark', 'Dark reference frame', [('All files', '*.*')], False),
            ('white', 'White reference frame', [('All files', '*.*')], False),
            ('rawdir', 'RawImages folder', None, True),
            ('flightlog', 'Flight-log CSV', [('CSV', '*.csv'), ('All files', '*.*')], False),
        ]
        for i, (key, label, filetypes, is_dir) in enumerate(rows):
            ttk.Label(pick_frame, text=label, width=24).grid(row=i, column=0, sticky='w', padx=4, pady=2)
            ttk.Entry(pick_frame, textvariable=self.paths[key], width=70).grid(
                row=i, column=1, sticky='we', padx=4, pady=2)
            ttk.Button(pick_frame, text='Browse…',
                       command=lambda k=key, ft=filetypes, d=is_dir: self._browse(k, ft, d)).grid(
                row=i, column=2, padx=4, pady=2)
        pick_frame.columnconfigure(1, weight=1)

        gain_frame = ttk.Frame(pick_frame)
        gain_frame.grid(row=len(rows), column=0, columnspan=3, sticky='w', padx=4, pady=4)
        ttk.Label(gain_frame, text='GAIN_WHITE:').pack(side='left')
        ttk.Entry(gain_frame, textvariable=self.gain_white, width=6).pack(side='left', padx=(2, 12))
        ttk.Label(gain_frame, text='GAIN_LAKE:').pack(side='left')
        ttk.Entry(gain_frame, textvariable=self.gain_lake, width=6).pack(side='left', padx=2)

        ttk.Button(pick_frame, text='Load References', command=self._on_load).grid(
            row=len(rows) + 1, column=0, columnspan=3, pady=6)

        time_frame = ttk.LabelFrame(self, text='Time / Frame')
        time_frame.pack(fill='x', padx=8, pady=6)
        self.time_scale = ttk.Scale(time_frame, from_=0, to=1, variable=self.time_var,
                                     orient='horizontal', command=self._on_time_change, state='disabled')
        self.time_scale.pack(fill='x', padx=6, pady=6)
        self.time_label = ttk.Label(time_frame, text='Load references to enable time selection.')
        self.time_label.pack(anchor='w', padx=6)

        self.analyze_btn = ttk.Button(time_frame, text='Analyze This Frame',
                                       command=self._on_analyze, state='disabled')
        self.analyze_btn.pack(pady=4)

        plot_frame = ttk.LabelFrame(self, text='Preview')
        plot_frame.pack(fill='both', expand=True, padx=8, pady=6)
        self.fig = Figure(figsize=(9, 3.5))
        self.ax_img, self.ax_spec = self.fig.subplots(1, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.export_btn = ttk.Button(self, text='Export CSV', command=self._on_export, state='disabled')
        self.export_btn.pack(pady=4)

        status_frame = ttk.LabelFrame(self, text='Status')
        status_frame.pack(fill='both', padx=8, pady=6)
        self.status_text = tk.Text(status_frame, height=8, state='disabled', wrap='word')
        self.status_text.pack(fill='both', padx=4, pady=4)

    def _log(self, msg):
        self.status_text.configure(state='normal')
        self.status_text.insert('end', msg + '\n')
        self.status_text.see('end')
        self.status_text.configure(state='disabled')

    def _browse(self, key, filetypes, is_dir):
        if is_dir:
            path = filedialog.askdirectory(title=f'Select {key}')
        else:
            path = filedialog.askopenfilename(title=f'Select {key}', filetypes=filetypes or [('All files', '*.*')])
        if path:
            self.paths[key].set(path)

    def _on_load(self):
        try:
            self._load_references()
        except Exception as e:
            messagebox.showerror('Load failed', str(e))

    def _load_references(self):
        for key, var in self.paths.items():
            if not var.get():
                raise ValueError(f'Please select the {key} path first.')
        try:
            gain_white = float(self.gain_white.get())
            gain_lake = float(self.gain_lake.get())
        except ValueError:
            raise ValueError('GAIN_WHITE / GAIN_LAKE must be numbers.')

        xml_path = self.paths['xml'].get()
        dark_path = self.paths['dark'].get()
        white_path = self.paths['white'].get()
        rawdir = self.paths['rawdir'].get()
        flightlog = self.paths['flightlog'].get()

        cfg = parse_ocif_xml(xml_path)
        wls = cfg['wavelengths']
        self._log(f"Sensor {cfg['width']}x{cfg['height']} -> cube "
                   f"{cfg['n_bands']} bands x {cfg['n_spatial']} spatial px")
        self._log(f'lambda {wls[0]:.1f}-{wls[-1]:.1f} nm')

        dark_frame, bits = load_raw_frame(dark_path, cfg)
        white_frame, _ = load_raw_frame(white_path, cfg)
        dark_cube = bin_to_cube(dark_frame, cfg)
        white_cube = bin_to_cube(white_frame, cfg)
        self._log(f'Detected {bits}-bit storage.')

        wmax = white_frame.max()
        cap = next(c for c in (255, 1023, 4095, 65535) if wmax <= c)
        sat_frac = 100 * (white_frame >= cap).mean()
        if sat_frac < 1.0:
            sat_bias = 1.0
            self._log(f'White max DN={wmax:.0f}, cap={cap}, {sat_frac:.2f}% saturated -> SAT_BIAS=1.0')
        else:
            sat_bias = 3.5
            self._log(f'White max DN={wmax:.0f}, cap={cap}, {sat_frac:.2f}% saturated -> SAT_BIAS=3.5 (stop-gap)')

        white_cube_cal = (white_cube - dark_cube) * (gain_lake / gain_white) + dark_cube
        off_panel_threshold = white_cube_cal.mean() * 0.4
        panel = white_cube_cal.mean(axis=0) > off_panel_threshold
        denom = np.where(np.abs(white_cube_cal - dark_cube) < 0.5, np.nan, white_cube_cal - dark_cube)
        self._log(f'Panel mask: {int(panel.sum())}/{len(panel)} spatial pixels kept.')

        gt, glat, glon = [], [], []
        with open(flightlog, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                try:
                    gt.append(sod_ampm(row['CUSTOM.updateTime [local]']))
                    glat.append(float(row['OSD.latitude']))
                    glon.append(float(row['OSD.longitude']))
                except Exception:
                    pass
        if not gt:
            raise ValueError('No valid rows parsed from flight-log CSV.')
        gt = np.array(gt)
        order = np.argsort(gt)
        gt = gt[order]
        self._log(f'GPS rows: {len(gt)} span {gt.min():.0f}-{gt.max():.0f}s')

        frame_times, frame_paths = [], []
        for name in sorted(os.listdir(rawdir)):
            if name.startswith('.') or name.lower().endswith('.xml'):
                continue
            full = os.path.join(rawdir, name)
            if not os.path.isfile(full):
                continue
            try:
                t = sod_from_frame_filename(name)
            except Exception:
                continue
            frame_times.append(t)
            frame_paths.append(full)
        if not frame_times:
            raise ValueError('No parseable frame files found in RawImages folder.')
        order = np.argsort(frame_times)
        frame_times = np.array(frame_times)[order]
        frame_paths = [frame_paths[i] for i in order]
        self._log(f'RawImages: {len(frame_paths)} frames span '
                   f'{frame_times.min():.0f}-{frame_times.max():.0f}s')

        if gt.max() < 3600:
            hour_offset = (int(frame_times[0]) // 3600) * 3600
            gt = gt + hour_offset
            self._log(f'Flight log used MM:SS format; recovered hour offset {hour_offset}s from first frame.')

        tmin = max(gt.min(), frame_times.min())
        tmax = min(gt.max(), frame_times.max())
        if tmin >= tmax:
            raise ValueError('Flight-log time range and RawImages frame times do not overlap.')

        self.cfg = cfg
        self.dark_cube = dark_cube
        self.denom = denom
        self.panel = panel
        self.sat_bias = sat_bias
        self.frame_times = frame_times
        self.frame_paths = frame_paths

        self.time_scale.configure(from_=tmin, to=tmax, state='normal')
        self.time_var.set((tmin + tmax) / 2)
        self._on_time_change(None)
        self._log(f'Ready. Time range {fmt_hms(tmin)} - {fmt_hms(tmax)}.')

    def _on_time_change(self, _value):
        if self.frame_times is None:
            return
        t = self.time_var.get()
        idx = int(np.argmin(np.abs(self.frame_times - t)))
        self.matched_idx = idx
        matched_t = self.frame_times[idx]
        name = Path(self.frame_paths[idx]).name
        self.time_label.configure(
            text=f'Selected: {fmt_hms(t)}   ->   Frame #{idx} ({name})   '
                 f'@ {fmt_hms(matched_t)}   Δt={abs(matched_t - t) * 1000:.0f} ms')
        self.analyze_btn.configure(state='normal')
        self.export_btn.configure(state='disabled')

    def _on_analyze(self):
        try:
            self._analyze_frame()
        except Exception as e:
            messagebox.showerror('Analyze failed', str(e))

    def _analyze_frame(self):
        idx = self.matched_idx
        path = self.frame_paths[idx]
        frame, _ = load_raw_frame(path, self.cfg)
        cube = bin_to_cube(frame, self.cfg)
        with np.errstate(divide='ignore', invalid='ignore'):
            R = (cube - self.dark_cube) / self.denom * PANEL_R / self.sat_bias
        R[:, ~self.panel] = np.nan

        wls = self.cfg['wavelengths']
        self.current_R = R
        self.current_wls = wls
        self.current_frame_name = Path(path).stem

        self.ax_img.clear()
        self.ax_spec.clear()
        finite = R[np.isfinite(R)]
        vmax = np.nanpercentile(R, 98) if finite.size else 1.0
        self.ax_img.imshow(np.clip(np.nan_to_num(R), 0, vmax), aspect='auto', cmap='viridis',
                            extent=[0, self.cfg['n_spatial'], wls[-1], wls[0]])
        self.ax_img.set_title('Reflectance')
        self.ax_img.set_xlabel('across-track px')
        self.ax_img.set_ylabel('nm')
        self.ax_spec.plot(wls, np.nanmean(R, axis=1), color='#1D9E75')
        self.ax_spec.set_title('Mean spectrum')
        self.ax_spec.set_xlabel('nm')
        self.ax_spec.set_ylabel('reflectance')
        self.ax_spec.grid(alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

        self._log(f'Analyzed frame #{idx} ({self.current_frame_name}); median R={np.nanmedian(R):.4f}')
        self.export_btn.configure(state='normal')

    def _on_export(self):
        if self.current_R is None:
            return
        default_name = f'{self.current_frame_name}_reflectance.csv'
        out_path = filedialog.asksaveasfilename(defaultextension='.csv', initialfile=default_name,
                                                 filetypes=[('CSV', '*.csv')])
        if not out_path:
            return
        try:
            n_spatial = self.current_R.shape[1]
            with open(out_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['wavelength_nm'] + [f'pixel_{i}' for i in range(n_spatial)])
                for wl, row in zip(self.current_wls, self.current_R):
                    writer.writerow([f'{wl:.3f}'] + [f'{v:.6f}' if np.isfinite(v) else '' for v in row])
            self._log(f'Exported: {out_path}')
        except Exception as e:
            messagebox.showerror('Export failed', str(e))


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
