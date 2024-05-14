#!/usr/bin/python3
import numpy as np
from general_sim_utils import import_h5_data, find_h5_files, parse_params_from_h5_filename
from multiprocessing import Pool

#Results returned is a list of dictionaries, one for each h5 file. Dictionary contains filename, efficiency, and input parameter values for each flag

class multi_Sim_Analyser:
    def __init__(self, analysis_settings) -> None:
        self.analysis_swttings = analysis_settings
        
        self.flags = analysis_settings['flags']
        self.numprocs = analysis_settings['num_analysis_procs']
        self.in_pulse_end_time = analysis_settings['in_pulse_end_time']
        self.out_pulse_start_time = analysis_settings['out_pulse_start_time']
        self.results_file_nametag = analysis_settings['results_file_nametag']
        self.sim_directory = analysis_settings['sim_directory']
        
        self.args_list = []
        self.results = []
        self.h5filenames = []
        self.numfiles = 0
        

    def get_all_sim_files_in_dir(self):
        self.h5filenames = find_h5_files(self.sim_directory, self.results_file_nametag)
        self.numfiles = len(self.h5filenames)

    def add_sim_files_to_analyse(self, filenames):
        self.h5filenames = filenames
        self.numfiles = len(self.h5filenames)

    def init_pool(self):
        if self.numprocs == 0:
            poolsize = self.numfiles
        else:
            poolsize = min([self.numfiles, self.numprocs])
        self.analysers = Pool(poolsize)

    def init_args(self):
        for n in self.h5filenames:
            self.args_list.append([n, self.in_pulse_end_time, self.out_pulse_start_time])
        
    def parse_results_into_dict(self):
        for result in self.results:
            result.update(parse_params_from_h5_filename(result["Filename"], self.flags))

    def run_multi_Sim_Analysis(self):
        self.init_args()
        self.init_pool()
        self.results = self.analysers.map(calc_efficiency, self.args_list)
        if self.results == []:
            print("No analysis results for: ", self.args_list)
        self.analysers.close()
        self.analysers.join()
        self.parse_results_into_dict()


def calc_Emag(Data):
    EI = Data['1/EI'][:]
    ER = Data['1/ER'][:]
    return abs(ER + 1j*EI)

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def integrate_in_pulse_1D_sim(EMag, t, in_pulse_end_time):
    t_idx_end = find_nearest_idx(t, in_pulse_end_time)
    x_index = 0
    return np.sum(pow(EMag[:t_idx_end, x_index], 2))


def integrate_out_pulse_1D_sim(EMag, t, out_pulse_start_time):
    t_idx_start = find_nearest_idx(t, out_pulse_start_time)
    x_index = len(EMag[1,:])-1
    return np.sum(pow(EMag[t_idx_start:, x_index], 2))
       

def calc_efficiency(args):
    filename = args[0]
    in_pulse_end_time = args[1]
    out_pulse_start_time = args[2]
    try:
        f = import_h5_data(filename)
    except:
        print("Error importing", filename)
        return 0

    Emag = calc_Emag(f)
    t = f['1/t'][:]
    in_power = integrate_in_pulse_1D_sim(Emag, t, in_pulse_end_time)
    out_power = integrate_out_pulse_1D_sim(Emag, t, out_pulse_start_time)
    if (in_power > 0):
        efficiency = out_power/in_power
    else:
        efficiency = 0
    
    return {"Filename":filename,"Efficiency":efficiency}



def main():

    in_pulse_end_time = 1.37e-6
    out_pulse_start_time = 2.54e-6

    flags = ['dm', 'De', 'Om', 'Tin', 'tgap']

    M = multi_Sim_Analyser(flags, in_pulse_end_time, out_pulse_start_time, 10)
    M.get_all_sim_files_in_dir()
    M.run_multi_Sim_Analysis()
    print(M.results)
    
    #print(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
if __name__ == "__main__":
    main()
