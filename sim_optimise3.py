#!/usr/bin/python3


import numpy as np
import math
import pickle
from os import getcwd

def log_progress(str):
    try:
        with open("optimisation_log.txt", 'a') as f:
            f.writelines(str)
    except Exception as error:
        print("Unable to write progrss to optimisation_log.txt")
        print(error)


def is_subset_Dicts(dict1, dict2):
    ##is dict1 a subset of dict2
    for key in list(dict1.keys()):
        if key not in dict2 or dict1[key] != dict2[key]:
            return False
    return True

def find_subset_Dict(dict_to_Find, dict_list):
    for d in dict_list:
        if is_subset_Dicts(dict_to_Find, d):
            return d
    return None

def fill_in_dict_list(list_to_fill, superset_list):
    for i in list_to_fill:
        matching = find_subset_Dict(i, superset_list)
        if matching != None:
            i.update(matching)
    return list_to_fill

def test_dicts():
    a = "a"
    b = "b"
    c = "c"
    h = [{a:1, b:2, c:3}, {a:4, b:5, c:2}, {a:1, b:6, c:3}]
    g = [{a:1, b:2}, {a:4, c:2}]
    print(g)
    print(h)
    print(fill_in_dict_list(g, h))
    print(h)
    f = []
    for entry in h:
        print("Entry:",entry.keys())
        if a in entry.keys():
            f.append(entry)
        print("Entry2:",entry)
    print(f)
    return 0

def find_dist(point1, point2, x, y):
    dx = point1[x] - point2[x]
    dy = point1[y] - point2[y]
    return math.sqrt(dx*dx+dy*dy)

def mod_sigmoid(x, limit):
  return (2*limit/(1 + math.exp(-2*x/limit))-limit)

def mod_flipped_sigmoid(x, steepness):
  return (-2/(1 + math.exp(-steepness*x)))+2

def find_crawl_dist(ref_point, cmp_point, crawl_speed, x, y, z):
    w_z = get_zdiff_weight(ref_point, cmp_point, z)
    w_dist =  get_dist_diff_weight(ref_point, cmp_point, crawl_speed, x, y)
    crawl_dist = crawl_speed*w_dist*w_z
    return crawl_dist

def get_zdiff_weight(ref_point, cmp_point, z):
    zdiff = (cmp_point[z] - (ref_point[z]+0.02))/(cmp_point[z] + (ref_point[z]+0.02))
    zdiff = (zdiff+1)/2
    if cmp_point[z]-ref_point[z] < 0:
        zdiff = 0
    #print("zdiff = ", zdiff)
    return zdiff

def get_dist_diff_weight(ref_point, cmp_point, crawl_speed, x, y):
    dist = find_dist(ref_point, cmp_point, x, y)
    norm_dist = mod_sigmoid(dist, crawl_speed)
    #print("dist = ", dist)
    #print("normdist = ", norm_dist)
    return norm_dist

def find_angle(from_point, to_point, x, y):
    dx = to_point[x]-from_point[x]
    dy = to_point[y]-from_point[y]
    if dx == 0 and dy == 0: 
        return 0
    else:
        return math.atan2(dy, dx)

def add_vectors_polar(dist1, ang1, dist2, ang2):
    new_x = dist1*math.cos(ang1)+dist2*math.cos(ang2)
    new_y = dist1*math.sin(ang1)+dist2*math.sin(ang2)
    new_dist = math.sqrt(new_x*new_x + new_y*new_y)
    if new_x == 0 and new_y == 0: 
        new_ang = 0
    else:
        new_ang = math.atan2(new_y, new_x)
    return {"length":new_dist, "angle":new_ang}
    

def do_crawl(ref_point, crawl_dist, crawl_angle, x, y):
    if crawl_dist == 0:
        crawl_dist = np.random.uniform(low=0, high=0.2)
        crawl_angle = np.random.uniform(low=-np.pi, high=np.pi)
    new_x = ref_point[x]+crawl_dist*math.cos(crawl_angle)
    new_y = ref_point[y]+crawl_dist*math.sin(crawl_angle)
    if new_x < 0:
        new_x = 0.01
    if new_y < 0:
        new_y = 0.01
    #print("point =", ref_point)
    #print("dist =", crawl_dist, "ang = ", crawl_angle)
    new_pt = {x:new_x, y:new_y}
    #print("new pt = ", new_pt)
    return new_pt


def crawl_optimise(current_points, all_points, crawl_speed, input_params, input_dims, z):
    
    if input_dims == 2:
        x = input_params[0]
        y = input_params[1]
    else:
        print("optimise 3 must have 2 input dimensions")
        return []
    num_points = len(all_points)
    new_pts = []
    if num_points > 0:
        for ref_point in current_points:
            crawl = {"length":0,"angle":0}
            for cmp_point in all_points:
                if cmp_point != ref_point:
                    angle = find_angle(ref_point, cmp_point, x, y)
                    
                    #print("_______________________")
                    #print("ref pt", ref_point)
                    #print("cmp pt", cmp_point) 
                    #print("Angle =", angle)
                    dist = find_crawl_dist(ref_point, cmp_point, crawl_speed, x, y, z)
                    #print("weighted dist =", dist)
                    #print("Old crawl = ", crawl)
                    crawl = add_vectors_polar(crawl["length"], crawl["angle"], dist, angle)
                    #print("New crawl = ", crawl)

            norm_crawl_len = mod_sigmoid(crawl["length"], 4*crawl_speed)*mod_flipped_sigmoid(ref_point[z], 4/crawl_speed)
            #print("norm_crawl_len =", norm_crawl_len)
            new_pts.append(do_crawl(ref_point, norm_crawl_len, crawl["angle"], x, y))     
    else:
        print("List of points is  empty")
    return new_pts


def save_state_pckl(filename, thing_to_save):
    dir = getcwd()
    flag = 'wb'   
    try:
        with open(dir+'/'+filename, flag) as f:
            try:
                pickle.dump(thing_to_save, f)
            except:
                print("Couldn't write state to", filename)
                return
    except:
        print("Couldn't write to", filename)
        return    
    return

def retrieve_state_pckl(filename):
    dir = getcwd()
    flag = 'rb'
    try:
        with open(dir+'/'+filename, flag) as f:
            try:
                state = pickle.load(f)
            except:
                print("Couldn't read state from", filename)
                return
    except:
        print("Couldn't write to", filename)
        return    
    return state


class Optimise3():
    def __init__(self, all_points, current_round, optimiser_settings) -> None:
        
        self.all_points = all_points
        self.round = current_round
        
        self.cost = optimiser_settings['cost']
        self.input_paramters = optimiser_settings['input_parameters']

        if ('optimisation_speed' in optimiser_settings.keys()):
            self.optimisation_speed = optimiser_settings['optimisation_speed']
        else:
            self.optimisation_speed = 0.85

        if ('input_dimensions' in optimiser_settings.keys()):
            self.input_dimensions = optimiser_settings['input_dimensions']
        else:
            self.input_dimensions = 2

        if ('maximise_cost' in optimiser_settings.keys()):
            self.maximise_cost = optimiser_settings['maximise_cost']
        else:
            self.maximise_cost = False
        
        if ('valid_cost_range' in optimiser_settings.keys()):
            self.valid_cost_range = optimiser_settings['valid_cost_range']
        else:
            self.valid_cost_range = [0,1]

        
        self.previous_points = []
        self.points_to_crawl = []
        self.ispointsinit = False
        self.new_inputs = []
        self.next_run = {
            'next_run_type':'points'
        }
    

    def remove_bad_input_pts(self):
        good_pts = []
        for pt in self.all_points:
            if self.valid_cost_range[0] > pt[self.cost] or pt[self.cost] > self.valid_cost_range[1]:
                pt[self.cost] = 0
            if pt not in good_pts:
                good_pts.append(pt)
        self.all_points = good_pts 

    def init_points(self):
        self.remove_bad_input_pts()
        if self.round > 1:
            self.previous_points = retrieve_state_pckl("points.pckl")
            current_points_unchecked = fill_in_dict_list(self.previous_points, self.all_points)
            self.points_to_crawl = []
            for point in current_points_unchecked:
                if self.cost in point.keys():
                    self.points_to_crawl.append(point)
        else:
            self.points_to_crawl = self.all_points
        self.ispointsinit = True


    def go(self):
        #if self.isinit():
            #raise self.err
        self.init_points()

        self.new_inputs = crawl_optimise(self.points_to_crawl, self.all_points, self.optimisation_speed, self.input_paramters, self.input_dimensions, self.cost)
        #print(self.new_inputs)
        save_state_pckl("points.pckl", self.new_inputs)
        self.next_run.update({'input_points':self.new_inputs})

        #for i in range(len(self.points_to_crawl)):
        #    print(self.points_to_crawl[i], "-->", self.new_inputs[i])

def main():

    test_dicts()


if __name__ == "__main__":
    main()

