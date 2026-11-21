"""
xmds_python_tools - run, analyse and optimise parameter sweeps of XMDS2
simulations.

The most commonly used pieces are re-exported here, so a driver script only
needs e.g.::

    from xmds_python_tools import sim_Optimisation_Sweeper, setup_run_subdir

Submodules
----------
general_sim_utils   compiling sims, run-dir setup, h5/CSV/pickle file helpers
run_sim_utils       multi_Sim_Runner: run a compiled sim over many parameter points
analysis_sim_utils  multi_Sim_Analyser: parse h5 output and evaluate the cost function
sim_optimisation    sim_Optimiser / sim_Optimisation_Sweeper: the optimisation loop
sim_optimiseN       OptimiseN: N-dimensional fruit-fly optimiser
run_optimal_points  CLI: rerun points from an optimal_points.csv
gem_visualiser      CLI: interactive viewer for GEM h5 output (needs matplotlib)
"""

from .general_sim_utils import (
    setup_run_subdir,
    compile_Sims,
    remove_unwanted_Files,
    print_res_to_CSV,
    find_h5_files,
    import_h5_data,
    parse_params_from_h5_filename,
)
from .run_sim_utils import multi_Sim_Runner
from .analysis_sim_utils import multi_Sim_Analyser, calc_efficiency
from .sim_optimiseN import OptimiseN
from .sim_optimisation import sim_Optimiser, sim_Optimisation_Sweeper

__all__ = [
    "setup_run_subdir",
    "compile_Sims",
    "remove_unwanted_Files",
    "print_res_to_CSV",
    "find_h5_files",
    "import_h5_data",
    "parse_params_from_h5_filename",
    "multi_Sim_Runner",
    "multi_Sim_Analyser",
    "calc_efficiency",
    "OptimiseN",
    "sim_Optimiser",
    "sim_Optimisation_Sweeper",
]
