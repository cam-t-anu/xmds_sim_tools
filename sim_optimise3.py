#!/usr/bin/python3


import numpy as np
import math
import pickle
from os import getcwd

from sympy import limit

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
    try:
        res = (2*limit/(1 + math.exp(-2*x/limit))-limit)
        return res
    except Exception as error:
        print("Error in mod_sigmoid with x =", x, "and limit =", limit)
        print(error)
        return 0

def mod_flipped_sigmoid(x, steepness):
  try:
      return (-2/(1 + math.exp(-steepness*x)))+2
  except Exception as error:
      print("Error in mod_flipped_sigmoid with x =", x, "and steepness =", steepness)
      print(error)
      return 0

def find_crawl_dist(ref_point, cmp_point, crawl_speed, x, y, z):
    try:
        w_z = get_zdiff_weight(ref_point, cmp_point, z)
        w_dist =  get_dist_diff_weight(ref_point, cmp_point, crawl_speed, x, y)
        crawl_dist = crawl_speed*w_dist*w_z
        return crawl_dist
    except Exception as error:        
        print("Error finding crawl distance between points:")
        print("ref point:", ref_point)
        print("cmp point:", cmp_point)
        print("crawl speed:", crawl_speed)
        print("x:", x, "y:", y, "z:", z)
        print(error)
        return 0

def get_zdiff_weight(ref_point, cmp_point, z):
    try:        
        zdiff = (cmp_point[z] - (ref_point[z]+0.02))/(cmp_point[z] + (ref_point[z]+0.02))
        zdiff = (zdiff+1)/2
        if cmp_point[z]-ref_point[z] < 0:
            zdiff = 0
        return zdiff
    except Exception as error:
        print("Error finding zdiff weight between points:")
        print("ref point:", ref_point)
        print("cmp point:", cmp_point)
        print("z:", z)
        print(error)
        return 0

def get_dist_diff_weight(ref_point, cmp_point, crawl_speed, x, y):
    try:
        dist = find_dist(ref_point, cmp_point, x, y)
        w_dist = mod_sigmoid(dist, crawl_speed)
        return w_dist
    except Exception as error:        
        print("Error finding distance weight between points:")
        print("ref point:", ref_point)
        print("cmp point:", cmp_point)
        print("crawl speed:", crawl_speed)
        print("x:", x, "y:", y)
        print(error)
    return 0


def find_angle(from_point, to_point, x, y):
    try:
        dx = to_point[x]-from_point[x]
        dy = to_point[y]-from_point[y]
        if dx == 0 and dy == 0: 
            return 0
        else:
            return math.atan2(dy, dx)
    except Exception as error:
        print("Error finding angle between points:")
        print("from point:", from_point)
        print("to point:", to_point)
        print("x:", x, "y:", y)
        print(error)
        return 0

def add_vectors_polar(dist1, ang1, dist2, ang2):
    try:
        new_x = dist1*math.cos(ang1)+dist2*math.cos(ang2)
        new_y = dist1*math.sin(ang1)+dist2*math.sin(ang2)
        new_dist = math.sqrt(new_x*new_x + new_y*new_y)
        if new_x == 0 and new_y == 0: 
            new_ang = 0
        else:
            new_ang = math.atan2(new_y, new_x)
        return {"length":new_dist, "angle":new_ang}
    except Exception as error:
        print("Error adding vectors in polar form:")
        print("dist1:", dist1, "ang1:", ang1)
        print("dist2:", dist2, "ang2:", ang2)
        print(error)
        return {"length":0, "angle":0}
    

def do_crawl(ref_point, crawl_dist, crawl_angle, x, y):
    try:
        if crawl_dist == 0:
            crawl_dist = np.random.uniform(low=0, high=0.2)
            crawl_angle = np.random.uniform(low=-np.pi, high=np.pi)
        new_x = ref_point[x]+crawl_dist*math.cos(crawl_angle)
        new_y = ref_point[y]+crawl_dist*math.sin(crawl_angle)
        if new_x < 0:
            new_x = 0.01
        if new_y < 0:
            new_y = 0.01
        new_pt = {x:new_x, y:new_y}
        return new_pt
    except Exception as error:
        print("Error doing crawl from point:", ref_point)
        print("crawl dist:", crawl_dist, "crawl angle:", crawl_angle)
        print("x:", x, "y:", y)
        print(error)
        return ref_point


def crawl_optimise(current_points, all_points, crawl_speed, input_params, input_dims, z):

    try:
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
                        dist = find_crawl_dist(ref_point, cmp_point, crawl_speed, x, y, z)
                        crawl = add_vectors_polar(crawl["length"], crawl["angle"], dist, angle)

                norm_crawl_len = mod_sigmoid(crawl["length"], 4*crawl_speed)*mod_flipped_sigmoid(ref_point[z], 4/crawl_speed)
                new_pts.append(do_crawl(ref_point, norm_crawl_len, crawl["angle"], x, y))     
        else:
            print("List of points is  empty")
        return new_pts
    except Exception as error:
        print("Error in crawl optimise with current points:", current_points)
        print("all points:", all_points)
        print("crawl speed:", crawl_speed)
        print("input params:", input_params)
        print("input dims:", input_dims)
        print("z:", z)
        print(error)
        return []


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
        try:
            good_pts = []
            for pt in self.all_points:
                if self.valid_cost_range[0] > pt[self.cost] or pt[self.cost] > self.valid_cost_range[1]:
                    pt[self.cost] = 0
                if pt not in good_pts:
                    good_pts.append(pt)
            self.all_points = good_pts 
        except Exception as error:
            print("Error occurred while removing bad input points:")
            print(error)

    def init_points(self):
        try:
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
        except Exception as error:
            print("Error occurred while initialising points for optimisation:")
            print(error)


    def go(self):
        try:
            self.init_points()
            self.new_inputs = crawl_optimise(self.points_to_crawl, self.all_points, self.optimisation_speed, self.input_paramters, self.input_dimensions, self.cost)
            save_state_pckl("points.pckl", self.new_inputs)
            self.next_run.update({'input_points':self.new_inputs})
        except Exception as error:
            print("Error occurred while running optimisation:")
            print(error)



def main():

    test_dicts()


if __name__ == "__main__":
    main()

