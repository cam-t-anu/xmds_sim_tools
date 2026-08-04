
import copy
import csv
import zlib
from math import floor, isclose, log10
from os import getcwd, path
import numpy as np
from run_sim_utils import multi_Sim_Runner
from analysis_sim_utils import multi_Sim_Analyser
from sim_optimiseN import OptimiseN
from general_sim_utils import print_res_to_CSV, compile_Sims, remove_unwanted_Files


def round_to_sig_figs(value, sig_figs=5):
    if value == 0:
        return 0.0
    return round(value, -int(floor(log10(abs(value)))) + (sig_figs - 1))


class sim_Optimiser:
    def __init__(self, inner_run_settings, analysis_settings, optimiser_settings, sim_dir) -> None:

        self.inner_run_settings = inner_run_settings
        self.analysis_settings = analysis_settings
        self.optimiser_settings = optimiser_settings
        self.sim_directory = sim_dir

        self.optimiser_version = self.optimiser_settings['optimiser_version']
        if self.optimiser_version not in [2,4]:
            print("currently optimiser versions are 2 and 4")
            raise ValueError
        elif self.optimiser_version == 4:
            self.Optimiser = OptimiseN
        
        self.rounds = self.optimiser_settings['optimisation_rounds']
        if self.rounds not in range(1,11):
            print("optimiser rounds not in [1,10]")
            raise ValueError

        self.round_results = []
        self.optimiser_results = []
        self.optimal_point = {}

        self.current_round = 0
        self.fixed_params = []

        self.initial_params_to_sweep = []

    def isInit(self):
        if (self.fixed_params != [] and self.flags != [] and self.optimisation_paramters != {} and self.sim_nametag != ''):
            return 1
        else: 
            return 0

    def do_Round_Sim_Run(self):
        self.MSR = multi_Sim_Runner(self.inner_run_settings)
        self.MSR.do_Run()

    def do_Round_Analsysis(self):
        self.MSA = multi_Sim_Analyser(self.analysis_settings)
        self.MSA.get_all_sim_files_in_dir()
        self.MSA.run_multi_Sim_Analysis()
        if len(self.MSA.results) > 0:
            self.round_results = self.MSA.results
        else:
            print("No results from analyser")
            print(self.analysis_settings)


    def do_Round_Optimisation(self):
        try:
            if len(self.round_results) == 0:
                print("No current round results to optimise, this will probably cause an error")
            Op = self.Optimiser(self.round_results, self.current_round, self.optimiser_settings)
            Op.go()
            self.inner_run_settings.update(Op.next_run)
            self.last_optimiser = Op
        except Exception as error:
            print("An error occured in optimisation round: ", self.current_round)
            print("Inputs: ", self.round_results)
            print("Settings: ", self.optimiser_settings)
            print("Error: ", error)


    def do_Round(self):
        self.current_round += 1
        print('Doing run', self.current_round)
        self.do_Round_Sim_Run()
        print('Doing analysis', self.current_round)
        self.do_Round_Analsysis()
        print('Doing optimisation', self.current_round)
        self.do_Round_Optimisation()

    def get_optimal_point(self):
        try: 
            self.optimal_point = sorted(self.optimiser_results, key=lambda x: x['Efficiency'], reverse=True)[0]
        except Exception as error:
            print("Error getting optimal point")
            print(error)

    def save_optimal_point(self):
        print_res_to_CSV("optimal_points.csv", self.optimal_point, self.sim_directory)

    def save_optimisation_results(self):
        print_res_to_CSV("optimisation_results.csv", self.optimiser_results, self.sim_directory)

    def do_Optimisation(self):
        for _ in range(self.rounds):
            self.do_Round()

        self.optimiser_results = self.round_results
        self.get_optimal_point()
        self.save_optimal_point()
        self.save_optimisation_results()
        if hasattr(self, 'last_optimiser'):
            self.last_optimiser.cleanup()


class sim_Optimisation_Sweeper:
    def __init__(self, outer_sweep_settings, inner_run_settings, analysis_settings, optimiser_settings, sim_dir = getcwd()) -> None:
        self.outer_sweep_settings = outer_sweep_settings
        self.inner_run_settings = inner_run_settings
        self.analysis_settings = analysis_settings
        self.optimiser_settings = optimiser_settings
        self.sim_directory = sim_dir

        self.outer_sweep_parameter = self.outer_sweep_settings['parameter']
        self.outer_sweep_values = self.outer_sweep_settings['values']
        
        if 'outer_routine' in self.outer_sweep_settings.keys():
            self.outer_routine = self.outer_sweep_settings['outer_routine']
            self.do_outer_routine = True
        else:
            self.outer_routine = None
            self.do_outer_routine = False

        if 'sim_xmds_filename' in self.outer_sweep_settings.keys():
            self.sim_xmds_filename = self.outer_sweep_settings['sim_xmds_filename']
        else:
            self.sim_xmds_filename = None

        # Optional: seed each outer step's initial round from a previous
        # optimal point instead of the fixed 'input_parameters' grid, on the
        # assumption that the optimum drifts smoothly across the outer sweep
        # (e.g. neighbouring 'De' values). The centre point comes from
        # 'bias_source_csv' if given (falling back to same-run carryover),
        # otherwise purely from the previous outer step run within this
        # process (see previous_optimal_point).
        self.carry_optimal_bias = self.outer_sweep_settings.get('carry_optimal_bias', False)
        self.optimal_bias_spread = self.outer_sweep_settings.get('optimal_bias_spread', 0.1)
        self.bias_source_csv = self.outer_sweep_settings.get('bias_source_csv', None)
        self.previous_optimal_point = None
        self._csv_bias_points = None

        self.prepare_sims()

    def prepare_sims(self):
        compile_Sims(dir=self.sim_directory, sim_file_name=self.sim_xmds_filename)
        remove_unwanted_Files(dir=self.sim_directory)

    def _csv_bias_point(self, outer_value):
        """
        Look up a previously-saved optimal point for *outer_value* (matched
        on the outer sweep parameter, e.g. De) from an external
        optimal_points.csv named by 'bias_source_csv'.

        Lets a fresh run (or one resuming after a crash) seed even its very
        first outer step's bias from a prior run's results, rather than
        requiring the carryover to happen live within the same process. The
        CSV is loaded once and cached; 'bias_source_csv' may be a specific
        path/filename, or True to mean this run's own (possibly
        still-being-written) 'optimal_points.csv'.
        """
        if self._csv_bias_points is None:
            filename = self.bias_source_csv if isinstance(self.bias_source_csv, str) else 'optimal_points.csv'
            self._csv_bias_points = self.read_optimal_points(filename)

        param = self.outer_sweep_parameter
        for point in self._csv_bias_points:
            if param in point and isclose(point[param], outer_value, rel_tol=1e-9, abs_tol=1e-9):
                return point
        return None

    def _center_point_for(self, outer_value):
        """Resolve the point to bias this outer step's initial round around: a CSV match takes priority, falling back to same-run carryover."""
        if self.bias_source_csv:
            csv_point = self._csv_bias_point(outer_value)
            if csv_point is not None:
                return csv_point
        return self.previous_optimal_point

    def _biased_initial_points(self, run_settings, center):
        """
        Build a fresh set of initial-round points by jittering *center*, one
        per combination in this step's 'input_parameters' grid (so the
        population size matches what a plain grid sweep would have
        produced).

        Falls back to None (meaning: use the plain grid sweep) if *center*
        doesn't cover every optimised parameter.
        """
        input_params = self.optimiser_settings['input_parameters']
        if not center or any(p not in center for p in input_params):
            return None

        bounds = self.optimiser_settings.get('param_bounds')
        grid = run_settings['input_parameters']
        n_points = 1
        for values in grid.values():
            n_points *= len(values)

        seed = self.optimiser_settings.get('seed')
        rng = np.random.default_rng([seed, zlib.crc32(b'carry_optimal_bias')]) if seed is not None else np.random.default_rng()

        points = []
        for _ in range(n_points):
            point = {}
            for p in input_params:
                has_bounds = bool(bounds and p in bounds)
                span = (bounds[p][1] - bounds[p][0]) if has_bounds else (abs(center[p]) if center[p] != 0 else 1.0)
                val = center[p] + rng.normal(0, self.optimal_bias_spread * span)
                if has_bounds:
                    lo, hi = bounds[p]
                    val = max(lo, min(hi, val))
                point[p] = round_to_sig_figs(val)
            points.append(point)
        return points

    def do_Optimisation_Sweep(self):
        if isinstance(self.outer_sweep_values, int) or isinstance(self.outer_sweep_values, float):
            self.outer_sweep_values = [self.outer_sweep_values]
        base_name = self.optimiser_settings.get('name', 'optimisation')
        for i in self.outer_sweep_values:
            run_settings = copy.deepcopy(self.inner_run_settings)
            try:
                run_settings['fixed_parameters'].update({self.outer_sweep_parameter:i})
                val_str = str(int(i)) if isinstance(i, float) and i == int(i) else str(i).replace('.', 'p')
                self.optimiser_settings['name'] = f"{base_name}_{self.outer_sweep_parameter}_{val_str}"

                if self.carry_optimal_bias:
                    center = self._center_point_for(i)
                    biased_points = self._biased_initial_points(run_settings, center)
                    if biased_points is not None:
                        run_settings['input_points'] = biased_points
                        run_settings['next_run_type'] = 'points'

                self.SimOpt = sim_Optimiser(run_settings, self.analysis_settings, self.optimiser_settings, self.sim_directory)
            except Exception as error:
                print("Error initialising sim_Optimiser")
                print("Outer sweep val:", i)
                print(run_settings, self.analysis_settings, self.optimiser_settings, self.sim_directory)
                print(error)
            else:
                try:
                    self.SimOpt.do_Optimisation()
                except Exception as error:
                    print("Error in optimisation run")
                    print(self.inner_run_settings, self.analysis_settings, self.optimiser_settings, self.sim_directory)
                    print(error)
                else:
                    if self.carry_optimal_bias and self.SimOpt.optimal_point:
                        self.previous_optimal_point = copy.deepcopy(self.SimOpt.optimal_point)

            if self.do_outer_routine:
                try:
                    self.outer_routine(self)
                except Exception as error:
                    print("Error in outer routine", self.outer_routine)
                    print(error)

            remove_unwanted_Files(dir=self.sim_directory, file_extensions=['xsil', 'h5', 'pckl'])

    def read_optimal_points(self, optimal_points_filename='optimal_points.csv', sig_figs=5):
        csv_path = path.join(self.sim_directory, optimal_points_filename)
        if not path.isfile(csv_path):
            print("No optimal points file found at", csv_path)
            return []

        param_keys = self.analysis_settings['flags']
        points = []
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                point = {}
                for key in param_keys:
                    value = row.get(key)
                    if value not in (None, ''):
                        try:
                            point[key] = round_to_sig_figs(float(value), sig_figs)
                        except ValueError:
                            point[key] = value
                points.append(point)
        return points

    def run_optimal_points(self, optimal_points_filename='optimal_points.csv', sig_figs=5):
        """
        Re-runs every optimal point recorded in optimal_points.csv.

        Each outer sweep step's optimal h5 result is wiped by
        remove_unwanted_Files before the next step starts, so this reads the
        saved parameters back out of the CSV and reruns them all in one
        batch, leaving the resulting h5 files in place for analysis.

        Parameters are rounded to sig_figs significant figures, since xmds
        encodes them verbatim in the output filename.
        """
        points = self.read_optimal_points(optimal_points_filename, sig_figs)
        if not points:
            print("No optimal points to run from", optimal_points_filename)
            return

        run_settings = copy.deepcopy(self.inner_run_settings)
        run_settings['fixed_parameters'] = {}
        run_settings['input_points'] = points
        run_settings['next_run_type'] = 'points'

        self.optimal_points_runner = multi_Sim_Runner(run_settings)
        self.optimal_points_runner.do_Run()



