
from os import getcwd
from run_sim_utils import multi_Sim_Runner
from analysis_sim_utils import multi_Sim_Analyser
from sim_optimise3 import Optimise3
from general_sim_utils import print_res_to_CSV, compile_Sims, remove_unwanted_Files


class sim_Optimiser:
    def __init__(self, inner_run_settings, analysis_settings, optimiser_settings, sim_dir) -> None:

        self.inner_run_settings = inner_run_settings
        self.analysis_settings = analysis_settings
        self.optimiser_settings = optimiser_settings
        self.sim_directory = sim_dir

        self.optimiser_version = self.optimiser_settings['optimiser_version']
        if self.optimiser_version not in [2,3]:
            print("currently optimiser versions are 2 and 3")
            raise ValueError
        elif self.optimiser_version == 3:
            self.Optimiser = Optimise3
        
        self.rounds = self.optimiser_settings['optimisation_rounds']
        if self.rounds not in range(1,11):
            print("optimiser rounds not in [1,10]")
            raise ValueError

        self.round_results = []
        self.optimiser_results = []
        self.optimal_point = {}

        self.current_round = 0
        self.fixed_params = []
        self.flags = []
        self.in_pulse_end_time = 0
        self.out_pulse_start_time = 0

        self.initial_params_to_sweep = []


    def add_fixed_params(self, fixed_params):
        self.fixed_params = fixed_params

    def add_initial_sweep_params(self, initial_params_to_sweep):
        self.initial_params_to_sweep = initial_params_to_sweep

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
        self.round_results = self.MSA.results

    def do_Round_Optimisation(self):
        #print("Analysis results:", self.round_results)
        Op = self.Optimiser(self.round_results, self.current_round, self.optimiser_settings)
        Op.go()
        #print(Op.next_run)
        self.inner_run_settings.update(Op.next_run)

    def do_Round(self):
        self.current_round += 1
        print('Doing run', self.current_round)
        self.do_Round_Sim_Run()
        print('Doing analysis', self.current_round)
        self.do_Round_Analsysis()
        print('Doing optimisation', self.current_round)
        self.do_Round_Optimisation()

    def get_optimal_point(self):
        self.optimal_point = sorted(self.optimiser_results, key=lambda x: x['Efficiency'], reverse=True)[0]

    def save_optimal_point(self):
        print_res_to_CSV("optimal_points.csv", self.optimal_point, self.sim_directory)

    def save_optimisation_results(self):
        print_res_to_CSV("optimisation_results.csv", self.optimiser_results, self.sim_directory)

    def do_Opimisation(self):
        for _ in range(self.rounds):
            self.do_Round()
        
        self.optimiser_results = self.round_results
        self.get_optimal_point()
        self.save_optimal_point()
        self.save_optimisation_results()


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
        self.prepare_sims()
        
    def prepare_sims(self):
        compile_Sims(dir=self.sim_directory)
        remove_unwanted_Files(dir=self.sim_directory)

    def do_Optimisation_Sweep(self):
        for i in self.outer_sweep_values:
            self.inner_run_settings['fixed_parameters'].update({self.outer_sweep_parameter:i})
            self.SimOpt = sim_Optimiser(self.inner_run_settings, self.analysis_settings, self.optimiser_settings, self.sim_directory)
            self.SimOpt.do_Opimisation()
            if self.do_outer_routine:
                self.outer_routine(self)
            remove_unwanted_Files(dir=self.sim_directory, file_extensions=['xsil', 'h5', 'pckl'])



