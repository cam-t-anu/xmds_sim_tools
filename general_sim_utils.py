from os import getcwd, path, listdir
from subprocess import run
import pickle
import csv
import h5py

def compile_Sims(dir=getcwd(), sim_file_names=None):
    if sim_file_names == None:
        for file in listdir(dir):
            if file.endswith(".xmds"):
                run(["xmds2", dir+"/"+file])
    else:
        for f in sim_file_names:
            run(["xmds2", dir+"/"+f])
    return

def remove_unwanted_Files(dir=getcwd(), file_extensions = ['cc']):
    #run("rm " + dir + "/*.xsil", shell=True)
    for e in file_extensions:
        run("rm " + dir + "/*."+ e, shell=True)
    return

def add_to_pckl_list_File(filename, thing_to_add, dir=getcwd()):
    if(path.isfile(dir+'/'+filename)):
        flag = 'ab'
    else:
        flag = 'wb'    
    try:
        with open(dir+'/'+filename, flag) as f:
            try:
                pickle.dump(thing_to_add, f)
            except:
                print("Couldn't write results to", filename)
                return
    except:
        print("Couldn't write to", filename)
        return    

    return


def print_res_to_CSV(filename, res, dir=getcwd()):
    if type(res) == dict:
        res = [res]
    if(path.isfile(dir+'/'+filename)):
        flag = 'a'
        try:
            with open(dir+'/'+filename, 'r') as f:
                csv_reader = csv.DictReader(f)
                keys = csv_reader.fieldnames
        except:
            print("Couldn't read header in ", dir+'/'+filename)
            keys = set().union(*(d.keys() for d in res))
        try:
            with open(dir+'/'+filename, flag) as f:
                try:
                    csv_writer = csv.DictWriter(f, keys)
                    
                except Exception as error:
                    print(res, keys)
                    print("Couldn't append results to", filename)
                    print(error)
                    return
                csv_writer.writerows(res)
        except:
            print("Couldn't write to existing ", filename)
            return
    else:
        flag = 'w'
        keys = set().union(*(d.keys() for d in res))
        try:
            with open(dir+'/'+filename, flag) as f:
                try:
                    csv_writer = csv.DictWriter(f, keys)
                    csv_writer.writeheader()
                    csv_writer.writerows(res)
                except:
                    print("Couldn't write results to", filename)
                    return
        except:
            print("Couldn't write to", filename)
            return    
    return


def find_h5_files(directory, sim_nametag):
    namelen = len(sim_nametag)
    file_list = []
    for file in listdir(directory):
        if (file[0:namelen] == sim_nametag and file[-3:] == '.h5'):
            file_list.append(file)

    if len(file_list) == 0:
        print("No results files matching", sim_nametag, "in dir", directory)

    return file_list

def import_h5_data(filename):
    return h5py.File(filename)

def parse_params_from_h5_filename(filename, flags):
    parsed = dict()
    
    for flag in flags:
        tmp_input_param_str = filename.split(flag)[1].split("_")[1].split(".")[:-1]

        if(len(tmp_input_param_str)>1):
            parsed.update([(flag,float(tmp_input_param_str[0]+'.'+tmp_input_param_str[1]))])
        else:
            parsed.update([(flag,float(tmp_input_param_str[0]))])
    return parsed