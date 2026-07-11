#!/usr/bin/python3

from subprocess import run, TimeoutExpired
from os import getcwd
from multiprocessing import Pool
from itertools import product
from time import sleep
from math import floor, log10


def round_to_sig_figs(value, sig_figs=5):
    """Round *value* to *sig_figs* significant digits (0 is returned unchanged)."""
    if value == 0:
        return 0.0
    return round(value, -int(floor(log10(abs(value)))) + (sig_figs - 1))


def check_param_conflicts(fixed_params, varying_keys):
    """
    Raise ValueError if any parameter name appears in both *fixed_params* and
    *varying_keys*. When a key is defined in both, the fixed value silently
    overwrites the varying value for every simulation (see add_param_sweep/
    add_points below), which shows up as a missing --arg and duplicate runs.
    """
    conflicts = sorted(set(fixed_params) & set(varying_keys))
    if conflicts:
        raise ValueError(
            f"Parameter(s) {conflicts} are set in both 'fixed_parameters' and the "
            "varying parameters for this run. The fixed value will silently "
            "override the varying value on every simulation, producing missing "
            "arguments and duplicate runs. Remove the key from one of the two."
        )


def run_Sim(simargs):
    cmd = []
    loglvl = None
    logfile = "/dev/null"
    for arg in simargs:
        if arg=="sim_name":
            cmd.insert(0, getcwd()+"/"+simargs[arg])
        elif arg=="timeout":
            timeout = simargs[arg]
        elif arg=="log":
            loglvl = simargs[arg]
        elif simargs[arg] != 'default':
            value = simargs[arg]
            if isinstance(value, float):
                value = round_to_sig_figs(value)
            cmd.append("--"+arg+"="+str(value))

    if(loglvl=="all" or loglvl=="timeouts_only"):
        logfile = ''
        for s in cmd:
            logfile = logfile+str(s)
        logfile =  getcwd() + '/' + logfile.replace(getcwd() + '/',"") + '.txt'
        logfile = logfile.replace("=", "_")
               
    try:
        with open(logfile, "w") as f:
            try:
                sleep(0.5)
                print(*cmd, flush=True)
                run(cmd, timeout=timeout, stdout=f, stderr=f)
            except TimeoutExpired:
                print("Command", cmd,"timed out after", timeout, "seconds")
            except:
                print("Something went wrong with", cmd)
            else:
                if loglvl=="timeouts_only":
                    run(["rm", logfile])
    except:
        print("Couldn't write to "+logfile)
    
    return


class sim_Pool:
    def __init__(self) -> None:
        self.pool_size = 1
        self.args = []
        self.isArgsInit = False
        self.isPoolInit = False
        self.isPoolRunning = False

    def init_args(self, args_array):
        self.args = args_array
        self.isArgsInit = True

    def init_pool(self, num_procs):
        if(num_procs > len(self.args)):
            self.pool_size = len(self.args)
        else:
            self.pool_size = num_procs
        try: 
            self.pool = Pool(self.pool_size)
        except Exception as error:
            print("Error initialising pool")
            print(error)
        else:
            self.isPoolInit = True

    def run_pool(self):
        if self.isPoolInit and self.isArgsInit:
            self.pool.map(run_Sim, self.args)
            self.isPoolRunning = True
        else:    
            print("Pool and/or args not initialised, can't run.")


    def busy_wait_and_close(self):
        if self.isPoolRunning:
            try:
                self.pool.close()
                self.pool.join()
            except:
                print("Error waiting for pool")
        else:
            print("No pool running")


class multi_Sim_Runner(sim_Pool):
    def __init__(self, inner_run_settings) -> None:
        super().__init__()
        self.is_init = False

        try:
            self.inner_run_settings = inner_run_settings
            self.fixed_parameters = self.inner_run_settings['fixed_parameters']
            self.sim_cmd_settings = self.inner_run_settings['sim_cmd_settings']

            self.num_procs = self.inner_run_settings['sim_procs_num']
            self.next_run_type = self.inner_run_settings['next_run_type']
            
            self.fixed_parameters_settings = self.fixed_parameters
            self.fixed_parameters_settings.update(self.sim_cmd_settings)

            if self.next_run_type == 'sweep':
                self.input_parameters = self.inner_run_settings['input_parameters']
                self.fixed_parameters.update(self.sim_cmd_settings)
                self.add_param_sweep(self.fixed_parameters_settings, self.input_parameters)
            elif self.next_run_type == 'points':
                self.input_points = self.inner_run_settings['input_points']
                self.add_points(self.fixed_parameters_settings, self.input_points)
        
            self.is_init = True
        except Exception as error:
            print("Error initialising the Multi_Sim_Runner")
            print(error)

    
    def add_param_sweep(self, fixed_params, params_to_sweep):
        try:
            check_param_conflicts(fixed_params, params_to_sweep.keys())
            keys, values = zip(*params_to_sweep.items())
            args_array = [dict(zip(keys, v)) for v in product(*values)]
            for d in args_array:
                d.update(fixed_params)
            super().init_args(args_array)
        except Exception as error:
            print("Error adding parameter sweep")
            print(error)
        return

    def add_points(self, fixed_params, points):
        try:
            if points:
                check_param_conflicts(fixed_params, points[0].keys())
            for point in points:
                point.update(fixed_params)
            super().init_args(points)
        except Exception as error:
            print("Error adding points")
            print(error)
        return

    def do_Run(self):
        if self.is_init:
            super().init_pool(self.num_procs)
            try: 
                super().run_pool()
            except Exception as error:
                print("Error from raised from pool")
                print(error)
            else:
                super().busy_wait_and_close()
        else:
            print("Runner not initialised properly, didn't run:")
            print(self.inner_run_settings)
        return






