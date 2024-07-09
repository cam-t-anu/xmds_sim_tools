#!/usr/bin/python3

from subprocess import run, TimeoutExpired
from os import getcwd
from multiprocessing import Pool
from itertools import product
from time import sleep


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
            cmd.append("--"+arg+"="+str(simargs[arg]))

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
        keys, values = zip(*params_to_sweep.items())
        args_array = [dict(zip(keys, v)) for v in product(*values)]
        for d in args_array:
            d.update(fixed_params)
        super().init_args(args_array)
        return

    def add_points(self, fixed_params, points):
        for point in points:
            point.update(fixed_params)
        super().init_args(points)
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






