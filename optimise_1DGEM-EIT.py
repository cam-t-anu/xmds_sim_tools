#!/usr/bin/python3

import time
from os import getcwd
import numpy as np
from sim_optimisation import sim_Optimisation_Sweeper


def outer_routine(args):
    print("Do this every outer sweep step!")

def do_sim_Run():

    tchar = 86.957e-09
    tmax = 57.5*tchar

    in_pulse_end_time = 0.35*tmax
    out_pulse_start_time = 0.52*tmax

    sim_directory = getcwd()

    outer_sweep_settings = {
        'parameter':        'a1',
        'values':           10,
        'outer_routine':    outer_routine, 
        'sim_xmds_filename': '1DGEM-EIT.xmds'
    }

    inner_run_settings = {
        'fixed_parameters':     {
            'Omg':  'default',
            'ds':   'default',
            'lds':  'default', 
            'ldw':  'default',
            'dr':   'default',
            'bias': 'default',
            'gr':   'default',
            'c0':   'default',
            'Pw':   'default',
            'Tin':  'default',
            'tgap': 'default',
            'Np': 'default',
            'a0': 'default',
            'am1': 'default',
            'pdiff': 'default',
            'pmdiff': 'default',
        },
        'input_parameters':     {
            'gft':   list(np.linspace(start=0.35, stop=0.42, num=3)), 
            'Ome':   list(np.linspace(start=2.1, stop=2.5, num=3))
        },
        'input_points': [],
        'sim_cmd_settings':     {
            "log":              "timeouts_only",
            "timeout":          20*60,
            "sim_name":         '1DGEM-EIT.out',
        },
        'sim_procs_num':    15,
        'next_run_type':    'sweep'
    }

    analysis_settings = {
        'flags':                ['De', 'Np', 'Ome', 'Omg', 'ds', 'dr', 'Pw', 'Tin', 'c0', 'tgap', 'a1', 'pdiff', 'pmdiff', 'gft', 'bias'],
        'in_pulse_end_time':    in_pulse_end_time,
        'out_pulse_start_time': out_pulse_start_time,
        'results_file_nametag': '1Dgem-eit',
        'num_analysis_procs':   5,
        'sim_directory':        sim_directory
    }

    optimiser_settings = {
        'name':                 '1dgemeit_optimisation',
        'optimisation_speed':   0.5, #0-1
        'noise_scale':          0.08,   # more aggressive exploration
        'optimiser_version':    3,
        'optimisation_rounds':  5,
        'cost':                 'Efficiency',
        'maximise_cost':        True,
        'valid_cost_range':     [0,1],
        'input_parameters':     ['gft','Ome'],
        'param_bounds':          {
            'gft': [0.35, 0.42],
            'Ome': [2.1, 2.5]
        },
        'input_dimensions':     2
    }

    S = sim_Optimisation_Sweeper(outer_sweep_settings, inner_run_settings, analysis_settings, optimiser_settings, sim_directory)
    S.do_Optimisation_Sweep()






def main():
    start = time.time()

    do_sim_Run()
    
    end = time.time()
    print("All done, time elapsed:", end-start, "seconds")
    exit()


if __name__ == "__main__":
    main()

