#!/usr/bin/python3
from os import getcwd
from pprint import pprint

import numpy as np
from general_sim_utils import import_h5_data, find_h5_files, parse_params_from_h5_filename
from multiprocessing import Pool

#Results returned is a list of dictionaries, one for each h5 file. Dictionary contains filename, efficiency, and input parameter values for each flag

class multi_Sim_Analyser:
    def __init__(self, analysis_settings) -> None:
        
        try:
            self.analysis_settings = analysis_settings
            
            self.flags = analysis_settings['flags']
            self.numprocs = analysis_settings['num_analysis_procs']
            self.in_pulse_end_time = analysis_settings.get('in_pulse_end_time', None)
            self.out_pulse_start_time = analysis_settings.get('out_pulse_start_time', None)
            self.results_file_nametag = analysis_settings['results_file_nametag']
            self.sim_directory = analysis_settings['sim_directory']
        except Exception as error:
            print("Error parsing analysis settings:")
            print(analysis_settings)
            print(error)

        self.args_list = []
        self.results = []
        self.h5filenames = []
        self.numfiles = 0
        self.is_init = False
        

    def get_all_sim_files_in_dir(self):
        self.h5filenames = find_h5_files(self.sim_directory, self.results_file_nametag)
        self.numfiles = len(self.h5filenames)
        if self.numfiles > 0:
            self.is_init = True
        else:
            print("No sim files matching", self.results_file_nametag, "found in" , self.sim_directory)

    def add_sim_files_to_analyse(self, filenames):
        self.h5filenames = filenames
        self.numfiles = len(self.h5filenames)
        if self.numfiles > 0:
            self.is_init = True

    def init_pool(self):
        if self.numprocs == 0:
            poolsize = self.numfiles
        else:
            poolsize = min([self.numfiles, self.numprocs])
        self.analysers = Pool(poolsize)

    def init_args(self):
        for n in self.h5filenames:
            self.args_list.append([n, self.in_pulse_end_time, self.out_pulse_start_time])
        
    def parse_results_into_list_of_dicts(self):
        try:
            for result in self.results:
                result.update(parse_params_from_h5_filename(result["Filename"], self.flags))
        except Exception as error:
            print("Error parsing results into list of dictionaries:")
            print(error)

    def run_multi_Sim_Analysis(self):
        if self.is_init == True:
            try:
                self.init_args()
                self.init_pool()
                self.results = self.analysers.map(calc_efficiency, self.args_list)
                self.analysers.close()
                self.analysers.join()
                if self.results == []:
                    print("No analysis results for: ", self.args_list)
                else:
                    self.parse_results_into_list_of_dicts()
            except Exception as error:
                print("Error raised during analysis of:")
                print(self.args_list)
                print(error)
        else:
            print("Analyser not initialised")
            print(self.analysis_settings)


def calc_Emag(Data):
    EI = Data['1/EI'][:]
    ER = Data['1/ER'][:]
    return abs(ER + 1j*EI)

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def detect_sim_dims(f):
    """Returns number of spatial dimensions: 1 for (t,x), 3 for (t,x,y,z)."""
    return len(f['1/ER'].shape) - 1

# --- 1D integration ---

def integrate_in_pulse_1D_sim(EMag, t, in_pulse_end_time):
    t_idx_end = find_nearest_idx(t, in_pulse_end_time)
    x_index = 0
    return np.sum(pow(EMag[:t_idx_end, x_index], 2))


def integrate_out_pulse_1D_sim(EMag, t, out_pulse_start_time):
    t_idx_start = find_nearest_idx(t, out_pulse_start_time)
    x_index = len(EMag[1,:])-1
    return np.sum(pow(EMag[t_idx_start:, x_index], 2))

# --- 3D integration (EMag shape: t, x, y, z) ---

def integrate_in_pulse_3D_sim(EMag, t, in_pulse_end_time):
    t_idx_end = find_nearest_idx(t, in_pulse_end_time)
    return np.sum(EMag[:t_idx_end, 0, :, :]**2)


def integrate_out_pulse_3D_sim(EMag, t, out_pulse_start_time):
    t_idx_start = find_nearest_idx(t, out_pulse_start_time)
    return np.sum(EMag[t_idx_start:, -1, :, :]**2)

# --- Auto-detection of pulse timing ---

def _input_power_timeseries(EMag):
    """Integrated |E|^2 at x=0 face as a function of time, for any dimensionality."""
    if EMag.ndim == 2:
        return EMag[:, 0]**2
    return np.sum(EMag[:, 0, :, :]**2, axis=(1, 2))


def _output_power_timeseries(EMag):
    """Integrated |E|^2 at x=-1 face as a function of time, for any dimensionality."""
    if EMag.ndim == 2:
        return EMag[:, -1]**2
    return np.sum(EMag[:, -1, :, :]**2, axis=(1, 2))


def auto_detect_pulse_times(EMag, t, threshold=0.01):
    """
    Estimate in_pulse_end_time and out_pulse_start_time from field data.

    in_pulse_end_time:    first time after the input-face peak where power
                          drops below threshold * peak.
    out_pulse_start_time: first time (after in_pulse_end_time) where the
                          output-face power rises above threshold * its max.

    threshold: fraction of peak power used as the on/off boundary (default 1%).
    Returns (in_pulse_end_time, out_pulse_start_time).
    """
    power_in  = _input_power_timeseries(EMag)
    power_out = _output_power_timeseries(EMag)

    # --- in_pulse_end_time ---
    peak_in_idx = int(np.argmax(power_in))
    peak_in = power_in[peak_in_idx]
    below = np.where(power_in[peak_in_idx:] < threshold * peak_in)[0]
    if len(below) > 0:
        in_end_idx = peak_in_idx + int(below[0])
    else:
        in_end_idx = len(t) // 2
    in_pulse_end_time = t[in_end_idx]

    # --- out_pulse_start_time ---
    # Find the peak of the output after the input ends, then search backward
    # from that peak — this avoids latching onto input leakthrough.
    power_out_after = power_out[in_end_idx:]
    peak_out_idx = in_end_idx + int(np.argmax(power_out_after))
    peak_out = power_out[peak_out_idx]
    before_peak = power_out[in_end_idx:peak_out_idx]
    below_before_peak = np.where(before_peak < threshold * peak_out)[0]
    if len(below_before_peak) > 0:
        out_pulse_start_time = t[in_end_idx + int(below_before_peak[-1])]
    else:
        out_pulse_start_time = t[in_end_idx]

    return in_pulse_end_time, out_pulse_start_time


def calc_efficiency(args):
    filename = args[0]
    in_pulse_end_time = args[1]
    out_pulse_start_time = args[2]
    try:
        f = import_h5_data(filename)
    except:
        print("Error importing", filename)
        return {"Filename": filename, "Efficiency": 0}

    try:
        Emag = calc_Emag(f)
        t = f['1/t'][:]
        n_dims = detect_sim_dims(f)

        if in_pulse_end_time is None or out_pulse_start_time is None:
            detected_in, detected_out = auto_detect_pulse_times(Emag, t)
            if in_pulse_end_time is None:
                in_pulse_end_time = detected_in
            if out_pulse_start_time is None:
                out_pulse_start_time = detected_out

        if n_dims == 1:
            in_power  = integrate_in_pulse_1D_sim(Emag, t, in_pulse_end_time)
            out_power = integrate_out_pulse_1D_sim(Emag, t, out_pulse_start_time)
        else:
            in_power  = integrate_in_pulse_3D_sim(Emag, t, in_pulse_end_time)
            out_power = integrate_out_pulse_3D_sim(Emag, t, out_pulse_start_time)

        efficiency = out_power / in_power if in_power > 0 else 0
        return {"Filename": filename, "Efficiency": efficiency,
                "in_pulse_end_time": in_pulse_end_time,
                "out_pulse_start_time": out_pulse_start_time}
    except Exception as error:
        print("Error calculating efficiency for", filename)
        print(error)
        return {"Filename": filename, "Efficiency": 0}



def main():

    sim_directory = getcwd()

    analysis_settings = {
        'flags':                ['De', 'Np', 'Omg', 'adyn', 'bias', 'a1', 'cba', 'cbo', 'cbsd', 'ds', 'gcb', 'temp'],
        'results_file_nametag': '3Dgem_ctrl',
        'num_analysis_procs':   5,
        'sim_directory':        sim_directory
    }

    M = multi_Sim_Analyser(analysis_settings)
    M.get_all_sim_files_in_dir()
    M.run_multi_Sim_Analysis()    
    pprint(sorted(M.results, key=lambda x: x['Efficiency'], reverse=True))
    
if __name__ == "__main__":
    main()
