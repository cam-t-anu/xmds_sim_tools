#!/usr/bin/python3
import sys
sys.path.append('/home/cam/Sims-Repo/XMDS2/new_GEM')
import time
import numpy as np
from sim_optimisation import sim_Optimisation_Sweeper


def outer_routine(args):
    print("Do this every outer sweep step!")

def do_sim_Run():

    tchar = 86.957e-09
    tmax = 45*tchar

    in_pulse_end_time = 0.35*tmax
    out_pulse_start_time = 0.65*tmax

    sim_directory = '/home/cam/Sims-Repo/XMDS2/new_GEM/EIT_GEM'

    #flags = ['De', 'Om', 'Pw','Tin','c0','c1','c2','c3','c4','dm','gf','lds','ldw','tgap']

    outer_sweep_settings = {
        'parameter':        'De',
        'values':           list(np.linspace(5,15,5)),
        'outer_routine':    outer_routine 
    }

    inner_run_settings = {
        'fixed_parameters':     {
            'lds':  'default', 
            'ldw':  'default',
            'gf':   'default',
            'c0':   'default',
            'c1':   'default',
            'c2':   'default',
            'c3':   'default',
            'c4':   'default',
            'Pw':   'default',
            'Tin':  'default',
            'tgap': 'default',
        },
        'input_parameters':     {
            'dm':   list(np.linspace(0.1, 8, 4)), 
            'Om':   list(np.linspace(0.1, 8, 4))
        },
        'input_points': [],
        'sim_cmd_settings':     {
            "log":              "timeouts_only",
            "timeout":          10*60,
            "sim_name":         '1DGEM-EIT.out',
        },
        'sim_procs_num':    10,
        'next_run_type':    'sweep'
    }

    analysis_settings = {
        'flags':                ['De', 'Om', 'Pw','Tin','c0','c1','c2','c3','c4','dm','gf','lds','ldw','tgap'],
        'in_pulse_end_time':    in_pulse_end_time,
        'out_pulse_start_time': out_pulse_start_time,
        'results_file_nametag': '1Dgem-eit',
        'num_analysis_procs':   5,
        'sim_directory':        sim_directory
    }

    optimiser_settings = {
        'optimiser_version':    3,
        'optimisation_rounds':  4,
        'cost':                 'Efficiency',
        'maximise_cost':        True,
        'valid_cost_range':     [0,1],
        'input_parameters':     ['dm','Om'],
        'input_dimensions':     2
    }

    S = sim_Optimisation_Sweeper(outer_sweep_settings, inner_run_settings, analysis_settings, optimiser_settings, sim_directory)
    S.do_Optimisation_Sweep()






def main():
    start = time.time()

    do_sim_Run()
    
    end = time.time()
    print("All done, time elapsed:", end-start, "seconds")
    exit


if __name__ == "__main__":
    main()

