# BioLector growth-curve plotting

Plots scattered-light (biomass) growth curves from **BioLector I** CSV exports and
estimates the maximum growth rate and lag time per condition.

Both BioLection export flavours are read automatically: the raw file
(`FILENAME;...` header, one row per reading) and the processed export
(`FILE NAME;...` header, `WELL No.;CONTENT;...` table with readings as columns,
decimal commas). The files in `data/` are the processed kind.

Experiment layout this is built for: 5 strains × 4 media × 5 replicates on 96-well
plates, one plate per strain except one plate that carries two strains. Blank wells
(CONTENT `B1`…`B4`, one code per medium) are not plotted.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Put the CSV files into `data/`.
2. Edit `config.json`:
   - `media`: CONTENT code → medium label. The default is X1 = oMLP 50 g/L,
     X2 = oMLP 0 g/L, X3 = oMLP pH 3, X4 = oMLP pH 8.
   - `files`: one entry per CSV with the file path and the strain name(s). Use forward
     slashes (`data/file.csv`), a backslash is not valid in JSON. For the plate with two
     strains list both, e.g. `"strains": ["WT", "BSA"]`: the first strain gets X1–X4,
     the second X5–X8.
   - `biomass_gain`: which Biomass filterset to use when the file has several gains.
     Gain 30 is set because gains 40 and 50 saturate in the growing wells (the script
     warns about saturated channels).
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

- **Signal**: the `Biomass` filterset with the configured gain. Raw files are corrected
  with the instrument reference (`amplitude × reference value / cycle reference`), the
  same as BioLection does; processed exports are already corrected.
- **Mean ± SD**: replicates are grouped per measurement cycle.
- **µmax** (1/h): steepest slope of ln(signal − background) in a sliding window of
  `window_hours` (default 2 h). Only windows with R² ≥ `r2_min` (0.95) and signal above
  5 % of the well's range are accepted; if no window qualifies the best one is used and
  the well is flagged.
- **Doubling time**: ln 2 / µmax.
- **Lag time** (h): tangent method, the time where the µmax tangent crosses the initial
  level (mean of the first `n_baseline` cycles).
- **Background** (`growth.background`): the scattered-light signal has a large constant
  offset (medium + plate, about 30 a.u. here) that must be removed before the log fit,
  otherwise µmax comes out far too low. Options:
  - `"fit"` (default): estimates the offset per well from the curve itself, as the
    constant that makes ln(signal − offset) most linear over the early growth phase.
    Needs no blanks.
  - `"blanks"`: mean signal of the blank wells of the same medium and file (B1 → medium 1,
    …). Not usable for the data in `data/`: the blank wells drift far above the samples.
  - `"initial"`: subtracts `initial_fraction` (default 0.9) of each well's own starting
    level; the remainder is taken as the inoculum signal.
  - a number, or a `{medium: number}` dict.

  µmax is fairly insensitive to the choice; the lag time is the least robust output
  because it depends on the small difference between the starting level and the offset.
  Compare the `flag` and `background` columns in the per-well table when in doubt.

## If the CONTENT codes are different

If every well has its own code (X1…X96) instead of one code per condition, give an explicit
map instead of `strains`:

```json
{"file": "data/plate.csv", "content_map": {"X1": ["Strain 1", "oMLP 50 g/L"], "X2": ["Strain 1", "oMLP 50 g/L"], "...": "..."}}
```

The script prints how many wells each code has, so a wrong mapping is visible immediately.
