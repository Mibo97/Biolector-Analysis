#!/usr/bin/env python3
"""Generate synthetic BioLector I CSV files in the raw BioLection export format.

They mimic the experiment layout of this repository: a 96-well plate, 4 media
(CONTENT X1..X4), 5 replicates each, 3 blanks (B1), and one plate that carries two
strains (X1..X4 and X5..X8). Used to test ``biolector_plot.py`` without real data.

    python examples/make_example_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def logistic(t, y0, ymax, mu, lag):
    """Simple growth curve with lag phase in scattered-light units."""
    t_eff = np.clip(t - lag, 0, None)
    return y0 + (ymax - y0) / (1 + ((ymax - y0) / (0.02 * ymax)) * np.exp(-mu * t_eff))


def write_file(path: Path, strains: dict[str, dict], n_cycles: int = 120, cycle_min: float = 10.0,
               n_rep: int = 5, n_blank: int = 3, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    media_params = {  # (mu 1/h, lag h, ymax)
        "oMLP 50 g/L": (0.45, 2.0, 60),
        "oMLP 0 g/L": (0.25, 4.0, 25),
        "oMLP pH 3": (0.15, 8.0, 18),
        "oMLP pH 8": (0.35, 3.0, 45),
    }
    media = list(media_params)
    ref_value = 179.0
    rows_letters = "ABCDEFGH"
    wells = [f"{r}{c:02d}" for r in rows_letters for c in range(1, 13)]

    # well layout: for each strain block, 4 media x 5 replicates, then blanks
    layout = []  # (well, content, strain, medium)
    w_i = 0
    for s_idx, (strain, sparams) in enumerate(strains.items()):
        for m_idx, medium in enumerate(media):
            code = f"X{m_idx + 1 + s_idx * len(media)}"
            for _ in range(n_rep):
                layout.append((wells[w_i], code, strain, medium))
                w_i += 1
    for _ in range(n_blank):
        layout.append((wells[w_i], "B1", None, None))
        w_i += 1

    lines = [
        f"FILENAME;\\Hard Disk2\\Result\\{path.name}",
        "PROTOCOL;synthetic_test",
        "FILE_VERSION;3.3;",
        "DATE START;2026-09-01;08:00:00",
        "DATE END;2026-09-02;04:00:00;;Last Reading;%d;Timestamp;0" % n_cycles,
        "DEVICE;BL000-SYNTH",
        "USER;test;COMMENT;synthetic data, not a real experiment",
        "PLATETYPE;MTP-96-Round;LOT;UNKOWN",
        "MTP ROWS;8",
        "MTP COLUMNS;12",
        "FILTERSETS;1;REFERENCE_MODE;2;MULTI_PMT;0;",
        "",
        "FILTERSET;FILTERNAME;EX [nm];EM [nm];LAYOUT;FILTERNR;GAIN;PHASESTATISTICSSIGMA;SIGNALQUALITYTOLERANCE;"
        "REFERENCE VALUE;EM2 [nm];GAIN2;PROCESS PARAMETER",
        f" 1;Biomass;620;620;96MTP;1;20;1.00;100.00;{ref_value:.2f};;;SET TEMPERATURE [°C];30.00",
        ";;;;;;;;;;;;SET HUMIDITY [rH];85.00",
        ";;;;;;;;;;;;SET O2 [%];20.95",
        ";;;;;;;;;;;;SET CO2 [%];0.00",
        ";;;;;;;;;;;;SET SHAKER FREQUENCY [rpm];1000.00",
        f";;;;;;;;;;;;SET CYCLE TIME [min];{cycle_min:.0f}",
        ";;;;;;;;;;;;SET EXP TIME [h];0",
        "",
        "READING;WELLNUM;CONTENT;DESCRIPTION;FILTERSET;TIME [h];AMPLITUDE;PHASE;ACT TEMP [°C];ACT HUMIDITY [rH];"
        "ACT O2 [%];ACT CO2 [%];COMMENTS;TEMP CHAMBER;TEMP TABLE;TEMP BOTTLE",
        "K;;;;;0.00465;;;;;;;Close Cover, Open Cover, ;",
    ]

    well_noise = {w: rng.normal(1.0, 0.04) for w, *_ in layout}
    dt_well = (cycle_min / 60) / (len(layout) + 5)
    for c in range(1, n_cycles + 1):
        t_cycle = (c - 1) * cycle_min / 60
        ref_amp = ref_value * rng.normal(1.0, 0.01)
        lines.append(f"R;;;;1;{t_cycle:.5f};{ref_amp:.2f};0.0;30.00;85.00;20.95;0.00;test measure for referencing ;30.0;30.0;30.0")
        for k, (well, code, strain, medium) in enumerate(layout):
            t = t_cycle + (k + 1) * dt_well
            if strain is None:
                val = 8.0 + rng.normal(0, 0.4)
            else:
                mu, lag, ymax = media_params[medium]
                f_mu, f_lag, f_ymax = strains[strain]["mu"], strains[strain]["lag"], strains[strain]["ymax"]
                val = logistic(t, 10.0, ymax * f_ymax, mu * f_mu, lag + f_lag) * well_noise[well] + rng.normal(0, 0.3)
            # store the "raw" amplitude the instrument would have written before reference correction
            raw = val * ref_amp / ref_value
            lines.append(f"C{c};{well};{code};;1;{t:.5f};{raw:.2f};0.0;30.00;85.00;20.95;0.00;;30.0;30.0;30.0")

    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    print(f"wrote {path} ({len(layout)} wells)")


if __name__ == "__main__":
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    write_file(out / "strain_A.csv", {"Strain A": {"mu": 1.0, "lag": 0.0, "ymax": 1.0}}, seed=1)
    write_file(out / "strain_B.csv", {"Strain B": {"mu": 0.8, "lag": 1.0, "ymax": 0.9}}, seed=2)
    write_file(out / "strain_C.csv", {"Strain C": {"mu": 1.2, "lag": -0.5, "ymax": 1.1}}, seed=3)
    write_file(out / "strain_D_E.csv", {"Strain D": {"mu": 0.6, "lag": 2.0, "ymax": 0.7},
                                        "Strain E": {"mu": 1.1, "lag": 0.5, "ymax": 1.2}}, seed=4)
