#!/usr/bin/env python3
"""Plot BioLector I scattered-light (biomass) growth curves and estimate growth parameters.

Usage
-----
    python biolector_plot.py                     # uses config.json
    python biolector_plot.py --config my.json    # other config
    python biolector_plot.py --log               # log-scaled y axis
    python biolector_plot.py --replicates        # extra QC PDF with every single replicate
    python biolector_plot.py --png               # also save the main figure as PNG

What it does
------------
1. Parses BioLector I CSV exports: the raw file (``FILENAME;...`` header, one row per
   reading) and the processed BioLection export (``FILE NAME;...`` header, wide table).
2. Picks the "Biomass" filterset (configurable gain). Raw files get the standard
   reference correction (``amplitude * reference_value / cycle_reference``).
3. Maps the CONTENT codes (X1, X2, ...) to strain + medium using ``config.json``.
   Blank wells (CONTENT ``B1``.. = one per medium) are not plotted. Other unmapped
   codes are ignored.
4. Plots one figure with one subplot per medium; every strain is a line (mean of
   the replicates) with a shaded +/- 1 SD band. Saved as PDF.
5. Estimates the maximum specific growth rate (sliding-window regression on the
   natural log of the signal) and the lag time (tangent method) for every well and
   writes per-well and per-condition summary tables as CSV.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

# Five categorical colors from a colorblind-validated palette (fixed order, never cycled).
STRAIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

CONTENT_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


# ----------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------
def _read_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def _to_float(series: pd.Series) -> pd.Series:
    """Convert a text column to float, accepting a decimal comma."""
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def parse_biolector1(path: Path) -> dict:
    """Parse a raw BioLector I CSV file.

    Returns a dict with ``metadata`` (dict), ``filtersets`` (DataFrame),
    ``references`` (DataFrame) and ``measurements`` (long DataFrame with columns
    cycle, well, content, filterset, time, amplitude).
    """
    lines = _read_lines(path)
    if not lines or not lines[0].startswith("FILENAME"):
        raise ValueError(
            f"{path.name}: does not look like a raw BioLector I file (first line should start with 'FILENAME;')."
        )

    sep = ";" if ";" in lines[0] else ","

    # --- split header / data
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("READING"):
            data_start = i
            break
    if data_start is None:
        raise ValueError(f"{path.name}: no 'READING;WELLNUM;...' table header found.")
    header_lines = lines[:data_start]

    # --- metadata (best effort, never fatal)
    metadata: dict = {}
    for line in header_lines:
        parts = [p.strip() for p in line.split(sep)]
        if len(parts) >= 2 and parts[0]:
            metadata[parts[0]] = parts[1]

    # --- filterset table
    fs_rows = []
    fs_cols = None
    for line in header_lines:
        if line.startswith("FILTERSET"):
            fs_cols = [c.strip() for c in line.split(sep)]
            continue
        if fs_cols is not None:
            parts = [p.strip() for p in line.split(sep)]
            if parts and parts[0].isdigit():
                fs_rows.append(dict(zip(fs_cols, parts)))
            elif not line.strip() or not line.startswith(";"):
                if fs_rows:
                    break
    if not fs_rows:
        raise ValueError(f"{path.name}: no FILTERSET definitions found in the header.")
    filtersets = pd.DataFrame(fs_rows)
    filtersets["FILTERSET"] = filtersets["FILTERSET"].astype(int)
    for col in ("GAIN", "REFERENCE VALUE"):
        if col in filtersets:
            filtersets[col] = _to_float(filtersets[col])

    # --- data table
    header = [c.strip() for c in lines[data_start].split(sep)]
    ncols = len(header)
    body = []
    for line in lines[data_start + 1 :]:
        if not line.strip():
            continue
        parts = line.split(sep)
        parts = (parts + [""] * ncols)[:ncols]  # pad / trim ragged rows
        body.append(parts)
    raw = pd.DataFrame(body, columns=header)

    def col(name_start: str) -> str:
        for c in raw.columns:
            if c.upper().startswith(name_start.upper()):
                return c
        raise ValueError(f"{path.name}: column starting with '{name_start}' not found. Columns: {list(raw.columns)}")

    c_read, c_well, c_content, c_fs, c_time, c_amp = (
        col("READING"), col("WELLNUM"), col("CONTENT"), col("FILTERSET"), col("TIME"), col("AMPLITUDE"),
    )
    raw[c_read] = raw[c_read].str.strip()

    # reference measurements ("R" rows): one per filterset per cycle
    ref = raw[raw[c_read] == "R"].copy()
    ref["filterset"] = pd.to_numeric(ref[c_fs], errors="coerce").astype("Int64")
    ref["time"] = _to_float(ref[c_time])
    ref["amplitude"] = _to_float(ref[c_amp])
    # assign cycle numbers to reference rows: the k-th R row of a filterset belongs to cycle k
    ref["cycle"] = ref.groupby("filterset").cumcount() + 1
    references = ref[["cycle", "filterset", "time", "amplitude"]].reset_index(drop=True)

    # measurements ("C<n>" rows)
    mes = raw[raw[c_read].str.match(r"^C\d+$")].copy()
    if mes.empty:
        raise ValueError(f"{path.name}: no measurement rows (READING = C1, C2, ...) found.")
    measurements = pd.DataFrame(
        {
            "cycle": mes[c_read].str[1:].astype(int),
            "well": mes[c_well].str.strip(),
            "content": mes[c_content].str.strip(),
            "filterset": pd.to_numeric(mes[c_fs], errors="coerce").astype(int),
            "time": _to_float(mes[c_time]),
            "amplitude": _to_float(mes[c_amp]),
        }
    ).reset_index(drop=True)

    return {"metadata": metadata, "filtersets": filtersets, "references": references, "measurements": measurements}


def parse_biolection_wide(path: Path) -> dict:
    """Parse a BioLection *processed* export (``FILE NAME;...`` header, ``WELL No.;CONTENT;...`` table).

    In this format every row is one well x channel and the readings are columns. Times come
    from the ``TIME [h] ->`` row. Numbers use a decimal comma. Channels named ``Blanked.N`` or
    ``Cal.*`` (BioLection's own blank subtraction / calibration) are skipped; only the raw
    filterset channels (1, 2, ...) are kept. The data are already reference corrected.
    """
    lines = _read_lines(path)
    sep = ";"

    table_start = None
    for i, line in enumerate(lines):
        if line.upper().startswith("WELL NO"):
            table_start = i
            break
    if table_start is None:
        raise ValueError(f"{path.name}: no 'WELL No.;CONTENT;...' table found.")
    header_lines = lines[:table_start]

    metadata: dict = {}
    for line in header_lines:
        parts = [x.strip() for x in line.split(sep)]
        if len(parts) >= 2 and parts[0]:
            metadata[parts[0]] = parts[1]

    # filterset table: header 'FILTERSET;FILTERNAME;...', then numbered rows
    fs_rows, fs_cols = [], None
    for line in header_lines:
        if line.startswith("FILTERSET;"):
            fs_cols = [c.strip() for c in line.split(sep)]
            continue
        if fs_cols is not None:
            parts = [x.strip() for x in line.split(sep)]
            if parts and parts[0].isdigit():
                fs_rows.append(dict(zip(fs_cols, parts)))
            elif fs_rows:
                break
    if not fs_rows:
        raise ValueError(f"{path.name}: no FILTERSET definitions found in the header.")
    filtersets = pd.DataFrame(fs_rows)
    filtersets["FILTERSET"] = filtersets["FILTERSET"].astype(int)
    for c in ("GAIN", "REFERENCE VALUE"):
        if c in filtersets:
            filtersets[c] = _to_float(filtersets[c])

    # time row
    times = None
    for line in lines[table_start + 1 : table_start + 6]:
        parts = line.split(sep)
        if len(parts) > 3 and parts[3].strip().upper().startswith("TIME"):
            times = _to_float(pd.Series(parts[4:])).to_numpy()
            break
    if times is None:
        raise ValueError(f"{path.name}: no 'TIME [h] ->' row found under the well table.")

    records = []
    for line in lines[table_start + 1 :]:
        parts = line.split(sep)
        if len(parts) < 5:
            continue
        well, content, _desc, channel = (x.strip() for x in parts[:4])
        if not well or not channel.isdigit():
            continue  # time/reading rows, ACT.TEMP rows, Blanked.* and Cal.* channels
        vals = _to_float(pd.Series(parts[4 : 4 + len(times)])).to_numpy()
        n = min(len(vals), len(times))
        for k in range(n):
            if np.isfinite(vals[k]) and np.isfinite(times[k]):
                records.append((k + 1, well, content, int(channel), times[k], vals[k]))
    if not records:
        raise ValueError(f"{path.name}: no measurement rows found in the well table.")
    measurements = pd.DataFrame(records, columns=["cycle", "well", "content", "filterset", "time", "amplitude"])
    references = pd.DataFrame(columns=["cycle", "filterset", "time", "amplitude"])
    return {"metadata": metadata, "filtersets": filtersets, "references": references, "measurements": measurements}


def parse_biolector_file(path: Path) -> dict:
    """Detect the export flavour and parse it."""
    first = _read_lines(path)[0].upper().replace(" ", "") if path.stat().st_size else ""
    if first.startswith("FILENAME;"):
        text = path.read_bytes()[:20000].decode("latin-1").upper()
        if "WELL NO." in text and "READING;WELLNUM" not in text:
            return parse_biolection_wide(path)
        return parse_biolector1(path)
    raise ValueError(f"{path.name}: unknown format (first line: {first[:40]!r}). Expected a BioLector I export.")


def saturation_report(m: pd.DataFrame, filtersets: pd.DataFrame) -> None:
    """Warn about channels whose values pile up at a ceiling (detector saturation)."""
    for fs_num, g in m.groupby("filterset"):
        v = g["amplitude"].dropna().to_numpy()
        if len(v) == 0:
            continue
        frac = float(np.mean(v >= 0.98 * v.max()))
        row = filtersets[filtersets["FILTERSET"] == fs_num]
        gain = row["GAIN"].iloc[0] if not row.empty and "GAIN" in row else np.nan
        if frac > 0.02:
            print(f"  WARNING: filterset {fs_num} (gain {gain:g}) looks saturated: "
                  f"{frac:.0%} of the values sit at the ceiling ({v.max():.1f}). Do not use this gain.")


def biomass_signal(parsed: dict, gain: float | None = None, reference_correction: bool = True) -> pd.DataFrame:
    """Return the scattered-light signal as a long DataFrame (well, content, cycle, time, value)."""
    fs = parsed["filtersets"]
    bio = fs[fs["FILTERNAME"].str.strip().str.lower().str.startswith("biomass")]
    if bio.empty:
        raise ValueError(f"No 'Biomass' filterset in file. Filtersets: {fs['FILTERNAME'].tolist()}")
    if gain is not None and "GAIN" in bio:
        sel = bio[np.isclose(bio["GAIN"], gain)]
        if sel.empty:
            raise ValueError(f"No Biomass filterset with gain {gain}. Available gains: {bio['GAIN'].tolist()}")
        bio = sel
    if len(bio) > 1:
        print(f"  note: {len(bio)} Biomass filtersets (gains {bio['GAIN'].tolist()}), using gain "
              f"{bio.iloc[0]['GAIN']:g}. Set 'biomass_gain' in the config to choose another.")
    elif gain is not None:
        print(f"  using Biomass filterset with gain {bio.iloc[0]['GAIN']:g}")
    fs_num = int(bio.iloc[0]["FILTERSET"])
    ref_value = float(bio.iloc[0].get("REFERENCE VALUE", np.nan))

    m = parsed["measurements"]
    m = m[m["filterset"] == fs_num].copy()
    m["value"] = m["amplitude"]

    if reference_correction and np.isfinite(ref_value) and ref_value > 0:
        r = parsed["references"]
        r = r[r["filterset"] == fs_num].set_index("cycle")["amplitude"]
        if not r.empty:
            factor = ref_value / m["cycle"].map(r)
            # cycles without a reference keep their raw value
            m["value"] = m["amplitude"] * factor.fillna(1.0)
    return m[["well", "content", "cycle", "time", "value"]].reset_index(drop=True)


# ----------------------------------------------------------------------------------
# Condition mapping
# ----------------------------------------------------------------------------------
def build_content_map(entry: dict, media: dict) -> dict[str, tuple[str, str]]:
    """Return {content_code: (strain, medium)} for one file entry of the config.

    Either the entry provides an explicit ``content_map`` ({"X1": ["Strain", "Medium"], ...})
    or a list of ``strains``; then medium codes are assigned in blocks:
    strain 1 -> X1..Xn, strain 2 -> X(n+1)..X(2n), ... with n = number of media.
    """
    if "content_map" in entry:
        return {k: (v[0], v[1]) for k, v in entry["content_map"].items()}

    strains = entry.get("strains")
    if not strains:
        raise ValueError(f"Config entry for {entry.get('file')} needs 'strains' or 'content_map'.")
    codes = list(media.keys())
    prefixes_numbers = []
    for c in codes:
        mm = CONTENT_RE.match(c)
        if not mm:
            raise ValueError(f"Medium content code '{c}' must look like X1, X2, ...")
        prefixes_numbers.append((mm.group(1), int(mm.group(2))))
    prefix = prefixes_numbers[0][0]
    n = len(codes)
    cmap = {}
    for s_idx, strain in enumerate(strains):
        for (pfx, num), medium_code in zip(prefixes_numbers, codes):
            code = f"{prefix}{num + s_idx * n}"
            cmap[code] = (strain, media[medium_code])
    return cmap


# ----------------------------------------------------------------------------------
# Growth parameters
# ----------------------------------------------------------------------------------
def fit_background(t: np.ndarray, y: np.ndarray, n_grid: int = 200) -> float:
    """Estimate the constant signal offset (medium + plate) from the growth curve itself.

    The early part of the curve (from the start until the signal has covered half of its
    total increase) is modelled as ``offset + N0 * exp(mu * t)``. The offset is found by a
    grid search: the value for which ln(y - offset) is most linear in time (highest R^2).
    """
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 8:
        return 0.0
    order = np.argsort(t)
    t, y = t[order], y[order]
    # smooth a little to find the level and the half-rise point
    k = min(5, len(y))
    ys = np.convolve(y, np.ones(k) / k, mode="same")
    y_lo, y_hi = float(np.min(ys)), float(np.max(ys))
    if y_hi - y_lo <= 0:
        return 0.0
    i_min = int(np.argmin(ys))
    half = y_lo + 0.5 * (y_hi - y_lo)
    after = np.where(ys[i_min:] >= half)[0]
    i_end = i_min + (int(after[0]) if len(after) else len(ys) - i_min)
    seg_t, seg_y = t[i_min:i_end + 1], y[i_min:i_end + 1]
    if len(seg_t) < 6:
        return 0.0
    best_bg, best_r2 = 0.0, -np.inf
    for bg in np.linspace(0.0, 0.995 * float(np.min(seg_y)), n_grid):
        ln_y = np.log(seg_y - bg)
        slope, intercept = np.polyfit(seg_t, ln_y, 1)
        pred = slope * seg_t + intercept
        ss_res = float(np.sum((ln_y - pred) ** 2))
        ss_tot = float(np.sum((ln_y - ln_y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if slope > 0 and r2 > best_r2:
            best_bg, best_r2 = float(bg), r2
    return best_bg


def growth_parameters(t: np.ndarray, y: np.ndarray, window_hours: float = 2.0, min_points: int = 5,
                      r2_min: float = 0.95, n_baseline: int = 3, background: float = 0.0) -> dict:
    """Estimate mu_max (1/h), doubling time (h), lag time (h) and max signal for one well.

    mu_max: steepest slope of ln(y) in a sliding window of ``window_hours`` that has
            R^2 >= r2_min and only uses signal above 5 % of the well's range.
    lag time: tangent method - intersection of the mu_max tangent with ln(y0),
            y0 = mean of the first ``n_baseline`` cycles.
    """
    out = {"mu_max": np.nan, "doubling_time": np.nan, "lag_time": np.nan, "t_mu_max": np.nan,
           "r2": np.nan, "y0": np.nan, "y_max": np.nan, "flag": ""}
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok] - background
    if len(t) < min_points:
        out["flag"] = "too few points"
        return out
    order = np.argsort(t)
    t, y = t[order], y[order]

    y0 = float(np.mean(y[:n_baseline]))
    y_max = float(np.max(y))
    out["y0"], out["y_max"] = y0, y_max

    threshold = y.min() + 0.05 * (y_max - y.min())
    valid = y > max(threshold, 1e-9)
    ln_y = np.where(valid, np.log(np.where(valid, y, 1.0)), np.nan)

    best = None  # (slope, intercept, r2, t_center)
    best_any = None
    n = len(t)
    for i in range(n):
        j = np.searchsorted(t, t[i] + window_hours, side="right")
        if j - i < min_points:
            continue
        seg_t, seg_ln = t[i:j], ln_y[i:j]
        if np.isnan(seg_ln).any():
            continue
        slope, intercept = np.polyfit(seg_t, seg_ln, 1)
        pred = slope * seg_t + intercept
        ss_res = float(np.sum((seg_ln - pred) ** 2))
        ss_tot = float(np.sum((seg_ln - seg_ln.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        cand = (slope, intercept, r2, float(np.mean(seg_t)))
        if best_any is None or slope > best_any[0]:
            best_any = cand
        if r2 >= r2_min and (best is None or slope > best[0]):
            best = cand

    if best is None:
        if best_any is None:
            out["flag"] = "no valid window"
            return out
        best = best_any
        out["flag"] = f"R2 < {r2_min}"

    slope, intercept, r2, t_center = best
    if slope <= 0:
        out["flag"] = (out["flag"] + "; " if out["flag"] else "") + "no growth"
        return out
    out.update(mu_max=slope, doubling_time=np.log(2) / slope, t_mu_max=t_center, r2=r2)
    if y0 > 0:
        out["lag_time"] = (np.log(y0) - intercept) / slope
    else:
        out["flag"] = (out["flag"] + "; " if out["flag"] else "") + "y0 <= 0, lag undefined"
    return out


# ----------------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------------
def load_config(cfg_path: Path) -> dict:
    """Read the JSON config; tolerate Windows backslashes in paths (an invalid JSON escape)."""
    text = cfg_path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError as err:
        try:
            cfg = json.loads(text.replace("\\", "/"))
        except json.JSONDecodeError:
            raise SystemExit(
                f"{cfg_path} is not valid JSON: {err}. Use forward slashes in paths (data/file.csv) "
                f"and check for missing commas or quotes."
            ) from err
        print(f"note: backslashes in {cfg_path.name} were read as '/'. Please use forward slashes in paths.")
        return cfg


def resolve_data_path(entry_file: str, config_dir: Path) -> Path:
    """Accept 'data/x.csv', 'x.csv' (looked up in data/), or an absolute path."""
    path = Path(entry_file)
    candidates = [path] if path.is_absolute() else [config_dir / path, config_dir / "data" / path.name]
    for c in candidates:
        if c.exists():
            return c
    data_dir = config_dir / "data"
    found = sorted(p.name for p in data_dir.glob("*.csv")) if data_dir.exists() else []
    raise FileNotFoundError(
        f"{entry_file} not found. CSV files present in {data_dir}: {found or 'none'}. "
        f"Fix the 'file' entries in the config (use forward slashes, e.g. data/file.csv)."
    )


def load_all(config: dict, config_dir: Path) -> pd.DataFrame:
    media = config["media"]
    frames = []
    for entry in config["files"]:
        path = resolve_data_path(entry["file"], config_dir)
        print(f"Reading {path.name}")
        parsed = parse_biolector_file(path)
        saturation_report(parsed["measurements"], parsed["filtersets"])
        sig = biomass_signal(parsed, gain=config.get("biomass_gain"),
                             reference_correction=config.get("reference_correction", True))
        cmap = build_content_map(entry, media)

        counts = sig.groupby("content")["well"].nunique()
        mapped = {c: n for c, n in counts.items() if c in cmap}
        ignored = {c: n for c, n in counts.items() if c not in cmap}
        for code, (strain, medium) in cmap.items():
            n = mapped.get(code, 0)
            if n == 0:
                print(f"  WARNING: content code {code} ({strain} / {medium}) has no wells in this file")
            elif n == 1:
                print(f"  WARNING: content code {code} ({strain} / {medium}) has only 1 well - "
                      f"replicates need the same CONTENT code; use 'content_map' if each well has its own code")
            else:
                print(f"  {code}: {strain} / {medium}: {n} wells")
        if ignored:
            print(f"  ignored content codes (blanks etc.): {', '.join(f'{c} ({n} wells)' for c, n in ignored.items())}")

        blank_prefix = config.get("blank_prefix", "B")
        is_blank = sig["content"].str.upper().str.startswith(blank_prefix.upper()) & ~sig["content"].isin(cmap)
        sig = sig[sig["content"].isin(cmap) | is_blank].copy()
        sig["is_blank"] = ~sig["content"].isin(cmap)
        # blank code B<k> belongs to medium k (B1 -> first medium, ...); B5.. wrap around
        media_labels = list(media.values())

        def blank_medium(code: str) -> str:
            mm = CONTENT_RE.match(code)
            if mm and media_labels:
                return media_labels[(int(mm.group(2)) - 1) % len(media_labels)]
            return "blank"

        sig["strain"] = sig["content"].map(lambda c: cmap[c][0] if c in cmap else "blank")
        sig["medium"] = sig["content"].map(lambda c: cmap[c][1] if c in cmap else blank_medium(c))
        blank_codes = sorted(sig.loc[sig["is_blank"], "content"].unique(), key=lambda c: int(CONTENT_RE.match(c).group(2)) if CONTENT_RE.match(c) else 0)
        if blank_codes:
            print("  blanks: " + ", ".join(f"{c} -> {blank_medium(c)} ({sig[sig['content'] == c]['well'].nunique()} wells)" for c in blank_codes))
        sig["file"] = path.name
        frames.append(sig)
    return pd.concat(frames, ignore_index=True)


def plot_conditions(data: pd.DataFrame, media_order: list[str], strain_order: list[str], out_pdf: Path,
                    log_y: bool = False, title: str | None = None, band: str = "sd") -> None:
    n_media = len(media_order)
    ncols = 2 if n_media > 1 else 1
    nrows = int(np.ceil(n_media / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), sharex=True, sharey=True, squeeze=False)
    colors = {s: STRAIN_COLORS[i % len(STRAIN_COLORS)] for i, s in enumerate(strain_order)}

    for ax, medium in zip(axes.flat, media_order):
        sub = data[data["medium"] == medium]
        for strain in strain_order:
            ss = sub[sub["strain"] == strain]
            if ss.empty:
                continue
            # replicates measured at slightly different times within a cycle -> aggregate per cycle
            g = ss.groupby("cycle").agg(time=("time", "mean"), mean=("value", "mean"),
                                        sd=("value", "std"), n=("value", "count")).sort_values("time")
            spread = g["sd"] if band == "sd" else g["sd"] / np.sqrt(g["n"])
            ax.fill_between(g["time"], g["mean"] - spread, g["mean"] + spread, color=colors[strain], alpha=0.18, lw=0)
            ax.plot(g["time"], g["mean"], color=colors[strain], lw=1.8, label=strain)
        ax.set_title(medium, fontsize=11, loc="left")
        ax.grid(True, color="#e6e6e3", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if log_y:
            ax.set_yscale("log")
    for ax in axes.flat[n_media:]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Scattered light (a.u.)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if not handles:
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title or "Growth curves (mean of replicates, band = ±1 SD)", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def plot_replicates(data: pd.DataFrame, media_order: list[str], strain_order: list[str], out_pdf: Path,
                    log_y: bool = False) -> None:
    """QC figure: one page per strain, one subplot per medium, every replicate as a thin line."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for strain in strain_order:
            sub = data[data["strain"] == strain]
            if sub.empty:
                continue
            n_media = len(media_order)
            ncols = 2 if n_media > 1 else 1
            nrows = int(np.ceil(n_media / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), sharex=True, sharey=True, squeeze=False)
            for ax, medium in zip(axes.flat, media_order):
                ss = sub[sub["medium"] == medium]
                for i, (well, w) in enumerate(ss.groupby("well")):
                    w = w.sort_values("time")
                    ax.plot(w["time"], w["value"], lw=1.0, color=STRAIN_COLORS[i % len(STRAIN_COLORS)], label=well)
                ax.set_title(medium, fontsize=11, loc="left")
                ax.grid(True, color="#e6e6e3", lw=0.6)
                ax.set_axisbelow(True)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                if log_y:
                    ax.set_yscale("log")
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=7, frameon=False, ncol=2)
            for ax in axes.flat[n_media:]:
                ax.set_visible(False)
            for ax in axes[-1, :]:
                ax.set_xlabel("Time (h)")
            for ax in axes[:, 0]:
                ax.set_ylabel("Scattered light (a.u.)")
            fig.suptitle(f"{strain}: individual replicates", fontsize=12, x=0.02, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {out_pdf}")


def resolve_background(cfg_bg, data: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Background (medium scatter without cells) per (file, medium), subtracted before the log fit.

    ``cfg_bg`` can be: a number (same for all), a dict {medium: number}, ``"blanks"`` (mean signal
    of the blank wells of the same file, 0 if a file has none) or ``"initial"`` (per well: mean of
    the first cycles; handled inside growth_parameters, disables the lag estimate).
    """
    keys = data[~data["is_blank"]][["file", "medium"]].drop_duplicates().itertuples(index=False)
    keys = [(f, m) for f, m in keys]
    if cfg_bg is None or cfg_bg == 0 or cfg_bg == "none":
        return {k: 0.0 for k in keys}
    if isinstance(cfg_bg, (int, float)):
        return {k: float(cfg_bg) for k in keys}
    if isinstance(cfg_bg, dict):
        return {(f, m): float(cfg_bg.get(m, 0.0)) for f, m in keys}
    if cfg_bg == "blanks":
        out = {}
        for f, m in keys:
            blanks = data[(data["file"] == f) & data["is_blank"]]
            b = blanks[blanks["medium"] == m]["value"]
            note = ""
            if b.empty:
                b = blanks["value"]
                note = " (no blank for this medium, all blanks of the file used)"
            if b.empty:
                print(f"  WARNING: no blank wells in {f}; background set to 0 for the growth-rate fit")
                out[(f, m)] = 0.0
            else:
                out[(f, m)] = float(b.mean())
                print(f"  background for {m} in {f}: {out[(f, m)]:.2f}{note}")
        return out
    if cfg_bg == "initial":
        return {k: "initial" for k in keys}
    if cfg_bg == "fit":
        return {k: "fit" for k in keys}
    raise ValueError(f"Unknown growth.background setting: {cfg_bg!r}")


def compute_growth_tables(data: pd.DataFrame, cfg: dict, media_order: list[str], strain_order: list[str]):
    rows = []
    samples = data[~data["is_blank"]]
    backgrounds = resolve_background(cfg.get("background", "fit"), data)
    for (strain, medium, well, file), w in samples.groupby(["strain", "medium", "well", "file"]):
        bg = backgrounds[(file, medium)]
        t, y = w["time"].to_numpy(float), w["value"].to_numpy(float)
        n_baseline = cfg.get("n_baseline", 3)
        if bg == "initial":
            # subtract most of the starting level; the rest is taken as the inoculum signal
            frac = float(cfg.get("initial_fraction", 0.9))
            bg = frac * float(np.nanmean(y[np.argsort(t)][:n_baseline]))
        elif bg == "fit":
            bg = fit_background(t, y)
        res = growth_parameters(
            t, y, window_hours=cfg.get("window_hours", 2.0), min_points=cfg.get("min_points", 5),
            r2_min=cfg.get("r2_min", 0.95), n_baseline=n_baseline, background=float(bg),
        )
        rows.append({"strain": strain, "medium": medium, "well": well, "file": file, "background": float(bg), **res})
    per_well = pd.DataFrame(rows)
    per_well["strain"] = pd.Categorical(per_well["strain"], strain_order)
    per_well["medium"] = pd.Categorical(per_well["medium"], media_order)
    per_well = per_well.sort_values(["strain", "medium", "well"]).reset_index(drop=True)

    agg = per_well.groupby(["strain", "medium"], observed=True).agg(
        n=("well", "count"),
        mu_max_mean=("mu_max", "mean"), mu_max_sd=("mu_max", "std"),
        doubling_time_mean=("doubling_time", "mean"), doubling_time_sd=("doubling_time", "std"),
        lag_time_mean=("lag_time", "mean"), lag_time_sd=("lag_time", "std"),
        y_max_mean=("y_max", "mean"), y_max_sd=("y_max", "std"),
    ).reset_index()
    return per_well, agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json", help="path to the JSON config (default: config.json)")
    ap.add_argument("--log", action="store_true", help="log-scaled y axis")
    ap.add_argument("--replicates", action="store_true", help="also write a QC PDF with every replicate")
    ap.add_argument("--sem", action="store_true", help="shade the standard error of the mean instead of the SD")
    ap.add_argument("--png", action="store_true", help="additionally save the main figure as PNG (300 dpi)")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config {cfg_path} not found.", file=sys.stderr)
        return 1
    cfg = load_config(cfg_path)
    config_dir = cfg_path.resolve().parent

    data = load_all(cfg, config_dir)
    samples = data[~data["is_blank"]]
    media_order = list(cfg["media"].values())
    strain_order = []
    for entry in cfg["files"]:
        for s in (entry.get("strains") or [v[0] for v in entry.get("content_map", {}).values()]):
            if s not in strain_order:
                strain_order.append(s)

    out_dir = config_dir / cfg.get("output_dir", "plots")
    res_dir = config_dir / cfg.get("results_dir", "results")
    band = "sem" if args.sem else "sd"
    title = cfg.get("title") or f"Growth curves (mean of replicates, band = ±1 {'SEM' if args.sem else 'SD'})"
    formats = ["pdf"] + (["png"] if args.png else [])
    for fmt in formats:
        plot_conditions(samples, media_order, strain_order, out_dir / f"growth_curves.{fmt}", log_y=args.log, title=title, band=band)
    if args.replicates:
        plot_replicates(samples, media_order, strain_order, out_dir / "growth_curves_replicates.pdf", log_y=args.log)

    per_well, summary = compute_growth_tables(data, cfg.get("growth", {}), media_order, strain_order)
    res_dir.mkdir(parents=True, exist_ok=True)
    per_well.to_csv(res_dir / "growth_parameters_per_well.csv", index=False, float_format="%.4f")
    summary.to_csv(res_dir / "growth_parameters_summary.csv", index=False, float_format="%.4f")
    samples.drop(columns=["is_blank"]).to_csv(res_dir / "biomass_long.csv", index=False, float_format="%.4f")
    print(f"Wrote {res_dir / 'growth_parameters_per_well.csv'}")
    print(f"Wrote {res_dir / 'growth_parameters_summary.csv'}")

    flagged = per_well[per_well["flag"] != ""]
    if not flagged.empty:
        print(f"\n{len(flagged)} well(s) flagged in the growth-rate fit (see 'flag' column):")
        print(flagged[["strain", "medium", "well", "flag"]].to_string(index=False))

    pd.set_option("display.width", 200)
    print("\nGrowth parameters per condition (mean ± SD of replicates):")
    show = summary.copy()
    for base in ("mu_max", "doubling_time", "lag_time", "y_max"):
        show[base] = show.apply(lambda r: f"{r[base + '_mean']:.3f} ± {r[base + '_sd']:.3f}", axis=1)
    print(show[["strain", "medium", "n", "mu_max", "doubling_time", "lag_time", "y_max"]]
          .rename(columns={"mu_max": "mu_max (1/h)", "doubling_time": "td (h)", "lag_time": "lag (h)", "y_max": "max signal"})
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
