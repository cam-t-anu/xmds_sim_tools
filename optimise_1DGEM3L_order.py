#!/usr/bin/python3

import time
import numpy as np
from sim_optimisation import sim_Optimisation_Sweeper
from general_sim_utils import setup_run_subdir


def outer_routine(args):
    print("Do this every outer sweep step!")

def do_sim_Run():

    tchar = 86.957e-09
    tmax = 57.5*tchar

    in_pulse_end_time = 0.35*tmax
    out_pulse_start_time = 0.52*tmax

    sim_directory = setup_run_subdir('1DGEM3L_order.xmds')

    outer_sweep_settings = {
        'parameter':        'De',
        'values':           [7],
        'outer_routine':    outer_routine, 
        'sim_xmds_filename': '1DGEM3L_order.xmds'
    }

    inner_run_settings = {
        'fixed_parameters':     {
            'lds':  'default', 
            'ldw':  'default',
            'gr':   'default',
            'c0':   'default',
            'Pw':   'default',
            'Tin':  'default',
            'tgap': 'default',
            'Np': 'default',
            'pdiff': 'default',
            'pmdiff': 'default',
            'gft': 'default',
            'ds': 'default'
        },
        'input_parameters':     {
            'Omg':   list(np.linspace(start=0.5, stop=5, num=4)),
            'bias':  list(np.linspace(start=0.2, stop=1.6, num=4))
        },
        'input_points': [],
        'sim_cmd_settings':     {
            "log":              "timeouts_only",
            "timeout":          20*60,
            "sim_name":         '1DGEM3L_order.out',
        },
        'sim_procs_num':    16,
        'next_run_type':    'sweep'
    }

    analysis_settings = {
        'flags':                ['De', 'Np', 'Omg', 'ds', 'Pw', 'Tin', 'c0', 'tgap', 'a1', 'pdiff', 'pmdiff', 'gft', 'bias'],
        'in_pulse_end_time':    in_pulse_end_time,
        'out_pulse_start_time': out_pulse_start_time,
        'results_file_nametag': '1Dgem-order',
        'num_analysis_procs':   5,
        'sim_directory':        sim_directory
    }

    optimiser_settings = {
        'name':                 '1dgemeit_optimisation',
        'optimisation_speed':   0.4, #0-1
        'noise_scale':          0.05,   # more aggressive exploration
        'optimiser_version':    4,
        'seed':                 37,
        'optimisation_rounds':  10,
        'cost':                 'Efficiency',
        'valid_cost_range':     [0,1],
        'input_parameters':     ['Omg', 'bias'],
        'param_bounds':          {
            'Omg': [0.1, 6],
            'bias': [0.1, 2]
        }
    }

    S = sim_Optimisation_Sweeper(outer_sweep_settings, inner_run_settings, analysis_settings, optimiser_settings, sim_directory)
    S.do_Optimisation_Sweep()
    S.run_optimal_points()






def main():
    start = time.time()

    do_sim_Run()
    
    end = time.time()
    print("All done, time elapsed:", end-start, "seconds")
    exit()


if __name__ == "__main__":
    main()

