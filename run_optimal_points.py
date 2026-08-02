#!/usr/bin/python3

"""
Reruns every parameter point listed in an optimal_points.csv file (as
written by sim_Optimiser.save_optimal_point) through the compiled xmds2
binary, using the same multi_Sim_Runner pool the optimiser itself uses.
"""

import argparse
import csv
import shutil
from math import floor, log10
from os import chdir, makedirs, path

from run_sim_utils import multi_Sim_Runner

DEFAULT_FLAGS = ['De', 'Np', 'Omg', 'ds', 'Pw', 'Tin', 'c0', 'tgap', 'a1', 'pdiff', 'pmdiff', 'gft', 'bias']


def round_to_sig_figs(value, sig_figs=5):
    if value == 0:
        return 0.0
    return round(value, -int(floor(log10(abs(value)))) + (sig_figs - 1))


def read_points(csv_path, flags, sig_figs=5):
    points = []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            point = {}
            for key in flags:
                value = row.get(key)
                if value not in (None, ''):
                    try:
                        point[key] = round_to_sig_figs(float(value), sig_figs)
                    except ValueError:
                        point[key] = value
            points.append(point)
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='?', default='optimal_points.csv',
                         help="Path to the optimal_points.csv file to run (default: %(default)s)")
    parser.add_argument('--sim-dir', default='1DGEM3L_order_run',
                         help="Directory containing the compiled xmds binary (default: %(default)s)")
    parser.add_argument('--output-dir', default=None,
                         help="Directory to run the simulations in and write results/logs to. The compiled "
                              "binary is copied here if it isn't already present. Defaults to --sim-dir.")
    parser.add_argument('--sim-name', default='1DGEM3L_order.out',
                         help="Compiled xmds binary name (default: %(default)s)")
    parser.add_argument('--procs', type=int, default=27,
                         help="Number of simulations to run in parallel (default: %(default)s)")
    parser.add_argument('--timeout', type=int, default=20 * 60,
                         help="Per-simulation timeout in seconds (default: %(default)s)")
    parser.add_argument('--log', default='timeouts_only', choices=['all', 'timeouts_only', 'none'],
                         help="xmds run logging level (default: %(default)s)")
    parser.add_argument('--flags', nargs='+', default=DEFAULT_FLAGS,
                         help="CSV columns to treat as simulation input parameters (default: %(default)s)")
    args = parser.parse_args()

    csv_path = path.abspath(args.csv)
    points = read_points(csv_path, args.flags)
    if not points:
        print("No points found in", csv_path)
        return

    print(f"Loaded {len(points)} point(s) from {csv_path}")

    sim_dir = path.abspath(args.sim_dir)
    if args.output_dir:
        run_dir = path.abspath(args.output_dir)
        makedirs(run_dir, exist_ok=True)
        binary_dst = path.join(run_dir, args.sim_name)
        if not path.exists(binary_dst):
            shutil.copy(path.join(sim_dir, args.sim_name), binary_dst)
        print(f"Running in {run_dir} (binary copied from {sim_dir})")
    else:
        run_dir = sim_dir

    chdir(run_dir)

    run_settings = {
        'fixed_parameters': {},
        'sim_cmd_settings': {
            'log': args.log,
            'timeout': args.timeout,
            'sim_name': args.sim_name,
        },
        'input_points': points,
        'sim_procs_num': args.procs,
        'next_run_type': 'points',
    }

    runner = multi_Sim_Runner(run_settings)
    runner.do_Run()


if __name__ == '__main__':
    main()
