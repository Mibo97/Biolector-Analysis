# BioLector growth-curve plotting

Plots scattered-light (biomass) growth curves from raw **BioLector I** CSV exports and
estimates the maximum growth rate and lag time per condition.

Experiment layout this is built for: 5 strains × 4 media × 5 replicates on 96-well
plates, one plate per strain except one plate that carries two strains. Blank wells are
not plotted, but their mean level is used as the background for the growth-rate fit.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Copy the four raw CSV files (as written by the BioLection software, first line
   `FILENAME;...`) into `data/`.
2. Edit `config.json`:
   - `media`: CONTENT code → medium label. The default is X1 = oMLP 50 g/L,
     X2 = oMLP 0 g/L, X3 = oMLP pH 3, X4 = oMLP pH 8.
   - `files`: one entry per CSV with the file path and the strain name(s). For the
     plate with two strains list both, e.g. `"strains": ["Strain 4", "Strain 5"]`:
     the first strain gets X1–X4, the second X5–X8.
3. Run:

```bash
python biolector_plot.py                 # config.json in the current folder
python biolector_plot.py --log           # log-scaled y axis
python biolector_plot.py --sem           # shade SEM instead of SD
python biolector_plot.py --replicates    # extra QC PDF with every single replicate
python biolector_plot.py --png           # also save the main figure as PNG
```

Outputs:

| File | Content |
|---|---|
| `plots/growth_curves.pdf` | One subplot per medium, one line per strain (mean of replicates) with ±1 SD band |
| `plots/growth_curves_replicates.pdf` | Optional: every replicate as a single line, one page per strain |
| `results/growth_parameters_summary.csv` | Per condition: mean ± SD of µmax, doubling time, lag time, max signal |
| `results/growth_parameters_per_well.csv` | The same per well, with R² and a `flag` column for questionable fits |
| `results/biomass_long.csv` | The reference-corrected scattered-light data in long format |

The summary table is also printed to the terminal.

## Try it on the synthetic example

```bash
python examples/make_example_data.py
python biolector_plot.py --config examples/config.json --png --replicates
```

## How the numbers are computed

- **Signal**: the `Biomass` filterset, corrected with the instrument reference
  (`amplitude × reference value / cycle reference`), the same as BioLection does.
  If a file has two Biomass filtersets (two gains) set `biomass_gain` in the config.
- **Mean ± SD**: replicates are grouped per measurement cycle.
- **µmax** (1/h): steepest slope of ln(signal − background) in a sliding window of
  `window_hours` (default 2 h). Only windows with R² ≥ `r2_min` (0.95) and signal above
  5 % of the well's range are accepted; if no window qualifies the best one is used and
  the well is flagged.
- **Doubling time**: ln 2 / µmax.
- **Lag time** (h): tangent method, the time where the µmax tangent crosses the initial
  level (mean of the first `n_baseline` cycles).
- **Background** (`growth.background`): `"blanks"` (default) uses the mean signal of the
  blank wells of the same file (CONTENT codes starting with `B`); alternatives are a
  number, a `{medium: number}` dict, `"initial"` (subtract each well's own starting
  level; this disables the lag estimate), or `0`. Without background subtraction the
  offset in the scattered-light signal makes µmax come out too low.

## If the CONTENT codes are different

If every well has its own code (X1…X96) instead of one code per condition, give an explicit
map instead of `strains`:

```json
{"file": "data/plate.csv", "content_map": {"X1": ["Strain 1", "oMLP 50 g/L"], "X2": ["Strain 1", "oMLP 50 g/L"], "...": "..."}}
```

The script prints how many wells each code has, so a wrong mapping is visible immediately.
