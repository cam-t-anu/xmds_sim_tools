# xmds_sim_tools
Tools to help make xmds simulations more useful.

## Repository layout

```
xmds_sim_tools/
├── xmds_python_tools/          # the Python package
│   ├── __init__.py             # re-exports the main classes/functions
│   ├── general_sim_utils.py    # compiling sims, run-dir setup, h5/CSV/pickle helpers
│   ├── run_sim_utils.py        # multi_Sim_Runner: run a compiled sim over many parameter points
│   ├── analysis_sim_utils.py   # multi_Sim_Analyser: parse h5 output, evaluate the cost function
│   ├── sim_optimisation.py     # sim_Optimiser / sim_Optimisation_Sweeper: the optimisation loop
│   ├── sim_optimiseN.py        # OptimiseN: N-dimensional fruit-fly optimiser
│   ├── run_optimal_points.py   # CLI: rerun every point in an optimal_points.csv
│   └── gem_visualiser.py       # CLI: interactive viewer for GEM h5 output
├── xmds_scripts/               # example .xmds sims and the driver scripts that optimise them
├── pyproject.toml, setup.cfg, setup.py   # packaging
└── README.md
```

## Installation

Install the package in editable mode, so changes to the source take effect
without reinstalling:

```bash
pip install -e /path/to/xmds_sim_tools            # core (numpy, h5py)
pip install -e "/path/to/xmds_sim_tools[plot]"    # also install matplotlib for plotting/visualiser
```

You also need `xmds2` on your `PATH` to compile and run simulations.

If you can't or don't want to install it, add the repository root (the folder
that *contains* `xmds_python_tools/`) to `PYTHONPATH` instead:

```bash
export PYTHONPATH=/path/to/xmds_sim_tools:$PYTHONPATH
```

## Usage

### From Python

```python
from xmds_python_tools import sim_Optimisation_Sweeper, setup_run_subdir
```

The package top level exports `setup_run_subdir`, `compile_Sims`,
`remove_unwanted_Files`, `print_res_to_CSV`, `find_h5_files`, `import_h5_data`,
`parse_params_from_h5_filename`, `multi_Sim_Runner`, `multi_Sim_Analyser`,
`calc_efficiency`, `OptimiseN`, `sim_Optimiser` and `sim_Optimisation_Sweeper`.
Anything else can be imported from its submodule, e.g.
`from xmds_python_tools.sim_optimiseN import visualize_search_history`.

### From the command line

Installing the package adds these commands. Each one can also be run without
installing as `python -m xmds_python_tools.<module>`.

| Command | Equivalent | What it does |
|---|---|---|
| `xmds-analyse [sim_dir] [--csv [file]]` | `python -m xmds_python_tools.analysis_sim_utils` | Compute efficiency for every h5 file in a directory |
| `xmds-run-optimal-points [csv] --sim-dir ... --sim-name ...` | `python -m xmds_python_tools.run_optimal_points` | Rerun every point in an `optimal_points.csv` |
| `xmds-gem-visualiser [file.h5]` | `python -m xmds_python_tools.gem_visualiser` | Interactive viewer for 1D/3D GEM output |
| `xmds-optimiser-selftest` | `python -m xmds_python_tools.sim_optimiseN` | Run the optimiser on synthetic test functions |

Use `--help` on any of them for the full list of options.

Note: the modules use relative imports, so run them with `python -m` rather
than `python xmds_python_tools/<file>.py`.

## User Guide

### Prerequisites
* An xmds simulation with:
  * input parameters defined as arguments
  * `<arguments append_args_to_output_filename="yes">` so that the analysis can parse out the input arguments
  * hdf5 output files with all the required data to evaluate your cost function
  * See `xmds_scripts/1DGEM3L_new.xmds` for an example

### You will need to write
* A top level simulation running script (a "driver") that includes some or all of:
    * `outer_sweep_settings` - if you want to perform multiple optimisation runs over a range of values
    * `inner_run_settings` - parameters that help the framework run your simulation, including the initial input values
    * `analysis_settings` - help the analyser parse the hdf5 output filenames and evaluate the cost function
    * `optimiser_settings` - defines optimisation parameters and other helpful parameters for optimisation
    * See `xmds_scripts/optimise_1DGEM3L_new.py` for an example format
* The driver should sit in the same directory as its `.xmds` file and be run
  from that directory. `setup_run_subdir('<sim>.xmds')` copies the `.xmds` into
  a `<sim>_run/` subdirectory and moves into it, so all compiled binaries, h5
  output, logs and result CSVs end up there:

```bash
cd xmds_scripts
python optimise_1DGEM3L_new.py
```

### You will likely want to write
* Your own cost function calculator
  * this can currently be written into `xmds_python_tools/analysis_sim_utils.py`
  * otherwise you can use the efficiency calculation that is already implemented (`calc_efficiency`)
* A function to be evaluated at every outer sweep point
  * This is useful if you want to perform some calculation or operation after every optimisation
  * See `outer_sweep_settings['outer_routine']`

### You may want to write
* Your own optimiser
  * Currently the included optimiser is `OptimiseN` in `sim_optimiseN.py` (selected with `'optimiser_version': 4`)
  * The software is designed such that other optimisers are easy to implement
  * An optimiser must be a class, with support for the API calls in `sim_Optimiser.do_Round_Optimisation()`
  * Register it in `sim_Optimiser.__init__` (in `sim_optimisation.py`) under a new `optimiser_version`


## General things TODO
* Write more optimisation algorithms, machine learning would be sweet
* Write better user and developer documentation (lol)
* Improve error handling
