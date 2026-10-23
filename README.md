# Code and data for 'Distinct weak antisymmetric interactions shape human brain functions as probability fluxes'

[![arXiv](https://img.shields.io/badge/arXiv-2508.20961-b31b1b.svg)](https://arxiv.org/abs/2508.20961)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23530107.svg)](https://doi.org/10.5281/zenodo.23530107)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Figures-CC_BY_4.0-lightgrey.svg)](LICENSE-figures)
[![CI](https://github.com/yoshiysoh/code-brain-probability-flux/actions/workflows/ci.yml/badge.svg)](https://github.com/yoshiysoh/code-brain-probability-flux/actions/workflows/ci.yml)

> Yoshiaki Horiike and Shin Fujishiro,
> *Distinct weak antisymmetric interactions shape human brain functions as probability fluxes*,
> [arXiv:2508.20961](https://arxiv.org/abs/2508.20961) (2025).

The code coarse-grains task fMRI recordings of the Human Connectome Project
(HCP) into binary states of clustered brain regions, measures the
probability fluxes between these states and the entropy production rate,
and infers a kinetic Ising model with symmetric and antisymmetric
interactions from the transition rates.

## Installation

```bash
git clone https://github.com/yoshiysoh/code-brain-probability-flux.git
cd code-brain-probability-flux
mamba env create -f env.yml
mamba activate brain-probability-flux
pip install -e . --no-deps
plotly_get_chrome -y  # headless Chrome for the brain-surface panels
```

Tested with Python 3.14 on Linux (AMD Ryzen 9 7950X) and macOS (Apple
silicon); package versions are pinned in `env.yml`.

## Tests

```bash
pytest
```

`pytest` checks the pipeline on synthetic data, so it needs no HCP
recordings, and checks that each file in `data/` matches its SHA-256 hash
in `data/SHA256SUMS` (`shasum -a 256 -c SHA256SUMS` inside `data/` does the
same). GitHub Actions runs `pytest`, `ruff check` and `ruff format --check`
(79 columns) on every push and pull request.

## Repository layout

```
src/brain_probability_flux/  analysis package
  analyzer/                  PFA class (coarse-graining, fluxes, entropy
                             production), clustering, network inference
  dina/                      kinetic Ising model: rates, stationary state,
                             fluxes; network plots
  plotting.py                loaders of data/ and panel layouts shared by
                             the notebooks
  provenance.py              run directories and provenance attributes
scripts/                     data generation
notebook/                    one notebook per figure (read data/, write
                             figures/)
data/                        group-level results behind every panel (HDF5)
                             and their SHA-256 hashes
figures/                     unlabeled panels of every figure (PDF, SVG, PNG)
tests/                       pipeline and data checks
```

## Data

This study uses HCP data, originally collected as described in Van Essen
et al. (2013)[^vanessen], Smith et al. (2013)[^smith] and Barch et al.
(2013)[^barch], and preprocessed and parcellated as in Lynn et al.
(2021)[^lynn].

The HCP recordings are **not included** (3.7 GB). To regenerate the
results, obtain the data from the HCP after accepting the [WU-Minn HCP
Open Access Data Use Terms][hcp-terms], preprocess and
parcellate them (Schaefer 100 parcels[^schaefer]) as described in Lynn et
al. (2021), and place the 16 files `<RUN>_LR.mat` and `<RUN>_RL.mat`, for
`<RUN>` in `REST1`, `EMOTION`, `WM`, `SOCIAL`, `LANGUAGE`, `RELATIONAL`,
`GAMBLING` and `MOTOR`, in `data/HCP_data_task/`.

`data/` contains the group-level results that every figure panel is drawn
from, so the notebooks run without the recordings. These results are
derived from HCP data and are redistributed under the same [WU-Minn HCP
Open Access Data Use Terms][hcp-terms], as term 4 of those terms
requires; by using them you agree to those terms. Each file stores its parameters, the seed and its provenance (git
commit, package versions, BLAS library and threads, CPU, command line) as
attributes. Rerunning a script writes to a new directory
`data/<YYYY-MM-DD>/` and never overwrites the shipped files.

The parcel labels of the Schaefer et al. (2018) parcellation[^schaefer],
which assign each parcel to a Yeo network, are downloaded by nilearn
(`fetch_atlas_schaefer_2018`) on first use and cached in `~/nilearn_data`.

## Generating the data

The scripts read the HCP recordings unless noted. Run them from the
repository root; each writes `data/<YYYY-MM-DD>/<file>.h5`. Runtimes are
for an AMD Ryzen 9 7950X with 16 threads.

### `data/aggregates.h5`
- **Command:** `python scripts/compute_aggregates.py --seed 42`
- **Contents:** `parcel_correlation` (100, 100), the correlation between
  the parcels over all recordings; `settings/<name>/`, one group per
  coarse-graining (`threshold_0.70`, the 7 clusters of the main analysis;
  `threshold_0.70_order1` and `_order2`, binarized at the zeros of the
  first and second derivatives; `threshold_0.80`, `_0.72`, `_0.71`, 4 to 6
  clusters; `maxclust_4` to `_6`; `yeo7`, the Yeo networks), each with
  `cluster_inds`, the observed `unique_states` and their `probability`, and
  per task (`task0` = rest to `task7` = motor) the transition counts
  `source`, `target`, `count` between state indices, the dwell time
  `waiting_time` in each state, and the attribute `total_time` (s);
  `pc_grid/`, the flux on the grid of the first two PC scores (fig-s02).
- **Runtime:** 4 min

### `data/inferred_networks.h5`
- **Command:** `python scripts/infer_networks.py` (reads
  `data/aggregates.h5`, not the recordings; `--data-dir` selects another
  directory)
- **Contents:** one group per model, each with `beta_J` (8, 7, 7),
  `beta_h` (8, 7), `rate_const` and the final loss or residuals:
  `arrhenius` (least squares, fig-04), `arrhenius_spindep` (fig-s12),
  `glauber` (gradient descent, fig-s08, fig-s11, fig-s13, fig-s24), `glauber_spindep`
  (fig-s13) and `shared_network` (one network for all tasks, fig-s15,
  fig-s16).
- **Runtime:** 1 min

### `data/inference_bootstrap.h5`
- **Command:** `python scripts/bootstrap_inference.py --seed 42`
- **Contents:** `arrhenius/beta_J` and `glauber/beta_J` (8, 100, 7, 7),
  the couplings re-inferred from 100 resamples of the transitions of each
  task (fig-05, fig-s09).
- **Runtime:** 1 min

### `data/epr_bootstrap.h5`
- **Command:** `python scripts/bootstrap_epr.py --seed 42`
- **Contents:** `num_clusters` (2 to 10), `thresholds`, and `eprs` and
  `frac_observed_transitions` (9, 8, 100): the entropy production rate
  (nats/s) and the fraction of observed transitions of 100 resamples per
  task (fig-s17).
- **Runtime:** 4 min

### `data/asymmetry_bootstrap.h5`
- **Command:** `python scripts/bootstrap_asymmetry.py --seed 42`
- **Contents:** `num_clusters` (4, 5, 6), `ratio` (3, 8) and `ratio_boot`
  (3, 8, 100): the ratio of the Frobenius norms of the antisymmetric and
  symmetric couplings and its bootstrap (fig-s14).
- **Runtime:** 1 min

### `data/stationary_bootstrap.h5`
- **Command:** `python scripts/bootstrap_stationary.py --seed 42`
- **Contents:** `probability_boot` (8, 200, 128), the state distribution of
  200 resamples of the time points of each task (fig-s09).
- **Runtime:** 1 min

### `data/rate_constant_bootstrap.h5`
- **Command:** `python scripts/bootstrap_rate_constants.py --seed 42`
  (also reads `data/inferred_networks.h5`)
- **Contents:** groups `arrhenius` (fig-s12) and `glauber` (fig-s13),
  each with the spin-dependent rate constants `A_sd` (8, 7, 100) and, for
  the spin-independent (`SI`) and spin-dependent (`SD`) fits, the
  correlation `r_<fit>` and squared error `loss_<fit>` (8, 100) between
  the fitted and observed fluxes.
- **Runtime:** 3 min

To regenerate everything from the recordings:

```bash
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMBA_NUM_THREADS=16
RUN=data/$(date +%F)
python scripts/compute_aggregates.py
python scripts/infer_networks.py --data-dir $RUN
python scripts/bootstrap_inference.py
python scripts/bootstrap_epr.py
python scripts/bootstrap_asymmetry.py
python scripts/bootstrap_stationary.py
python scripts/bootstrap_rate_constants.py --data-dir $RUN
python scripts/hcp_example_panels.py
```

This takes about 20 min with 16 threads.

## Reproducing the figures

`figures/` holds the panels of every figure, without panel labels, as PDF
and SVG. Each figure has an ID, which names its notebook and its directory
in `figures/`: `fig-01` to `fig-06` are Figs. 1–6, `fig-s01` to `fig-s10`
are Extended Data Figs. 1–10, and `fig-s11` to `fig-s24` are
Supplementary Figs. 1–14.

Each notebook in `notebook/` writes the panels of one figure to
`figures/<ID>/` as SVG, PDF and PNG. Run one with
`jupyter nbconvert --to notebook --execute notebook/<figure>.ipynb`, from
the repository root. The notebooks read `data/`; to plot a fresh run, set
`DATA_DIR` in the first cell to its run directory.

| Figure | ID | Panels | Notebook or script | Output in `figures/` |
|---|---|---|---|---|
| Fig. 1 | `fig-01` | a–e | `notebook/fig-01.ipynb` | `fig-01/a–e` |
| Fig. 1 | `fig-01` | f–h | `scripts/hcp_example_panels.py` (needs HCP) | `fig-01/f–h` |
| Fig. 2 | `fig-02` | all | `notebook/fig-02.ipynb` | `fig-02/fig-02` |
| Fig. 3 | `fig-03` | a–d | `notebook/fig-03.ipynb` | `fig-03/a–d` |
| Fig. 4 | `fig-04` | a–i | `notebook/fig-04.ipynb` | `fig-04/a–i` |
| Fig. 5 | `fig-05` | a–f | `notebook/fig-05.ipynb` | `fig-05/a–f` |
| Fig. 6 | `fig-06` | all | `notebook/fig-06.ipynb` | `fig-06/fig-06` |
| Extended Data Fig. 1 | `fig-s01` | all | `notebook/fig-s01.ipynb` | `fig-s01/fig-s01` |
| Extended Data Fig. 2 | `fig-s02` | all | `notebook/fig-s02.ipynb` | `fig-s02/fig-s02` |
| Extended Data Fig. 3 | `fig-s03` | all | `notebook/fig-s03.ipynb` | `fig-s03/fig-s03` |
| Extended Data Fig. 4 | `fig-s04` | a–l | `notebook/fig-s04.ipynb` | `fig-s04/a–l` |
| Extended Data Fig. 5 | `fig-s05` | g–i | `notebook/fig-s05.ipynb` | `fig-s05/g–i` |
| Extended Data Fig. 5 | `fig-s05` | a–f | `scripts/hcp_example_panels.py` (needs HCP) | `fig-s05/a–f` |
| Extended Data Fig. 6 | `fig-s06` | a–f | `notebook/fig-s06.ipynb` | `fig-s06/a–f` |
| Extended Data Fig. 7 | `fig-s07` | a–f | `notebook/fig-s07.ipynb` | `fig-s07/a–f` |
| Extended Data Fig. 8 | `fig-s08` | a–i | `notebook/fig-s08.ipynb` | `fig-s08/a–i` |
| Extended Data Fig. 9 | `fig-s09` | a–f | `notebook/fig-s09.ipynb` | `fig-s09/a–f` |
| Extended Data Fig. 10 | `fig-s10` | all | `notebook/fig-s10.ipynb` | `fig-s10/fig-s10` |
| Supplementary Fig. 1 | `fig-s11` | all | `notebook/fig-s11.ipynb` | `fig-s11/fig-s11` |
| Supplementary Fig. 2 | `fig-s12` | a–c | `notebook/fig-s12.ipynb` | `fig-s12/a–c` |
| Supplementary Fig. 3 | `fig-s13` | a–c | `notebook/fig-s13.ipynb` | `fig-s13/a–c` |
| Supplementary Fig. 4 | `fig-s14` | a, b | `notebook/fig-s14.ipynb` | `fig-s14/a, b` |
| Supplementary Fig. 5 | `fig-s15` | a–f | `notebook/fig-s15-s16.ipynb` | `fig-s15/a–f` |
| Supplementary Fig. 6 | `fig-s16` | all | `notebook/fig-s15-s16.ipynb` | `fig-s16/fig-s16` |
| Supplementary Fig. 7 | `fig-s17` | a–e | `notebook/fig-s17.ipynb` | `fig-s17/a–e` |
| Supplementary Fig. 8 | `fig-s18` | a–d | `notebook/fig-s18.ipynb` | `fig-s18/a–d` |
| Supplementary Fig. 9 | `fig-s19` | all | `notebook/fig-s19.ipynb` | `fig-s19/fig-s19` |
| Supplementary Fig. 10 | `fig-s20` | a–l | `notebook/fig-s20.ipynb` | `fig-s20/a–l` |
| Supplementary Fig. 11 | `fig-s21` | g–i | `notebook/fig-s21.ipynb` | `fig-s21/g–i` |
| Supplementary Fig. 11 | `fig-s21` | a–f | `scripts/hcp_example_panels.py` (needs HCP) | `fig-s21/a–f` |
| Supplementary Fig. 12 | `fig-s22` | a–f | `notebook/fig-s22.ipynb` | `fig-s22/a–f` |
| Supplementary Fig. 13 | `fig-s23` | all | `notebook/fig-s23.ipynb` | `fig-s23/fig-s23` |
| Supplementary Fig. 14 | `fig-s24` | all | `notebook/fig-s24.ipynb` | `fig-s24/fig-s24` |

Each notebook runs in under 30 s; the brain-surface panels (fig-01 c, d,
fig-s04, fig-s20) need the headless Chrome installed above.

The panels that show one example recording (fig-01 f to h, and fig-s05 and
fig-s21 a to f) are individual-level data, so they are not drawn from `data/`:
`python scripts/hcp_example_panels.py` draws them from the HCP recordings.

In the flux and traffic diagrams, the width of a hypercube edge is
proportional to its flux or traffic, and the shaft of a biplot arrow
(top-right panel) has the width of a flux (or traffic) of 0.04/`scale`
s<sup>-1</sup>, where `scale` is the argument of `plot_pca_by_task`: 0.001
s<sup>-1</sup> in fig-02 and 0.002 s<sup>-1</sup> in fig-s10.

## Determinism

All random numbers derive from `numpy.random.SeedSequence(42)` or
`numpy.random.default_rng(42)`. The data in `data/` were produced with
OpenBLAS 0.3.34 on an AMD Ryzen 9 7950X with 16 threads; JAX ran on the
CPU. Rerunning every script on that machine reproduces every dataset bit
for bit. On other hardware or BLAS libraries the last digits may differ:
on macOS (Apple silicon, one thread), the least-squares couplings agree
with `data/inferred_networks.h5` within a relative 1e-13 and the
gradient-descent (Glauber) couplings within 2e-8.

The panels are drawn in Helvetica if matplotlib finds it, and otherwise in
Arial (from `mscorefonts`); the line rendering also depends on the
matplotlib version.

## Citation

If you use this code or data, please cite:

```bibtex
@misc{horiike_distinct_2025,
  title         = {Distinct weak antisymmetric interactions shape human
                   brain functions as probability fluxes},
  author        = {Horiike, Yoshiaki and Fujishiro, Shin},
  year          = {2025},
  eprint        = {2508.20961},
  archivePrefix = {arXiv},
  primaryClass  = {physics.bio-ph},
  doi           = {10.48550/arXiv.2508.20961},
}
```

and the archived software: Yoshiaki Horiike and Shin Fujishiro, *Code and data for:
Distinct weak antisymmetric interactions shape human brain functions as
probability fluxes*, Zenodo,
[doi:10.5281/zenodo.23530107](https://doi.org/10.5281/zenodo.23530107).
See also [`CITATION.cff`](CITATION.cff).

## License

- Code: [BSD 3-Clause](LICENSE).
- Figures and figure panels: [CC BY 4.0](LICENSE-figures).
- Data derived from the HCP: the [WU-Minn HCP Open Access Data Use
  Terms][hcp-terms]. Data were
  provided by the Human Connectome Project, WU-Minn Consortium (Principal
  Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by
  the 16 NIH Institutes and Centers that support the NIH Blueprint for
  Neuroscience Research; and by the McDonnell Center for Systems
  Neuroscience at Washington University.

[^vanessen]: D. C. Van Essen, S. M. Smith, D. M. Barch, T. E. J. Behrens,
    E. Yacoub, and K. Ugurbil, "The WU-Minn Human Connectome Project: An
    overview," *NeuroImage* **80**, 62–79 (2013).
    https://doi.org/10.1016/j.neuroimage.2013.05.041
[^smith]: S. M. Smith *et al.*, "Resting-state fMRI in the Human Connectome
    Project," *NeuroImage* **80**, 144–168 (2013).
    https://doi.org/10.1016/j.neuroimage.2013.05.039
[^barch]: D. M. Barch *et al.*, "Function in the human connectome:
    Task-fMRI and individual differences in behavior," *NeuroImage* **80**,
    169–189 (2013). https://doi.org/10.1016/j.neuroimage.2013.05.033
[^lynn]: C. W. Lynn, E. J. Cornblath, L. Papadopoulos, M. A. Bertolero,
    and D. S. Bassett, "Broken detailed balance and entropy production in
    the human brain," *Proc. Natl. Acad. Sci. USA* **118**, e2109889118
    (2021). https://doi.org/10.1073/pnas.2109889118
[^schaefer]: A. Schaefer *et al.*, "Local-global parcellation of the human
    cerebral cortex from intrinsic functional connectivity MRI," *Cereb.
    Cortex* **28**, 3095–3114 (2018). https://doi.org/10.1093/cercor/bhx179

[hcp-terms]: https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms
