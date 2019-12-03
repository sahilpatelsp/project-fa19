import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import networkx as nx

from student_utils import *
from gurobipy import *
"""
======================================================================
  Complete the following function.
======================================================================
"""
def subtourelim(model, where):
  if where == GRB.callback.MIPSOL:
    selected = []
    # make a list of edges selected in the solution
    vals = model.cbGetSolution(model.getVars())
    #selected = tuplelist((i,j) for i,j in model._vars.keys() if vals[i,j] > 0.5)
    # find the shortest cycle in the selected edge list
    # tour = subtour(selected)
    # if len(tour) < n:
    #   # add a subtour elimination constraint
    #   expr = 0
    #   for i in range(len(tour)):
    #     for j in range(i+1, len(tour)):
    #       expr += model._vars[tour[i], tour[j]]
    #   model.cbLazy(expr <= len(tour)-1)

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """


    G = adjacency_matrix_to_graph(adjacency_matrix)[0]
    spl = dict(nx.all_pairs_dijkstra_path_length(G))
    loc_len = len(list_of_locations)
    home_len = len(list_of_homes)
    start_index = list_of_locations.index(starting_car_location)

    w = {}
    for i in range(loc_len):
        for j in range(loc_len):
            if adjacency_matrix[i][j] == 'x':
                w[i,j] = 0
            else:
                w[i,j] = adjacency_matrix[i][j]
    home_dict = {}
    for h in range(home_len):
        home_dict[h] = list_of_locations.index(list_of_homes[h])


    dist = {}
    for h in range(home_len):
        for i in range(loc_len):
            dist[h, i] = spl[home_dict[h]][i]
    try:
        m = Model()
        #v variable for each location
        v = m.addVars(loc_len, vtype=GRB.BINARY, name="v")
        #e variable for each edge as a 1D array
        e = m.addVars(loc_len, loc_len, vtype=GRB.BINARY, name="e")
        d = m.addVars(home_len, loc_len, vtype=GRB.BINARY, name="d")

        # Set objective
        m.setObjective(((2/3) * e.prod(w)) + d.prod(dist), GRB.MINIMIZE)
        m.addConstr(v[start_index] == 1)
        m.addConstr(v.sum() <= loc_len)
        m.addConstrs(e[i,j] == 0 for i in range(loc_len) for j in range(loc_len) if w[i,j] == 0)

        m.addConstrs((v[i] == 1) >> (e.sum('*', i) == e.sum(i, '*')) for i in range(loc_len))
        m.addConstrs((v[i] == 1) >> (e.sum('*', j) >= 1) for j in range(loc_len))
        m.addConstrs((v[i] == 1) >> (e.sum(i, '*') >= 1) for i in range(loc_len))

        m.addConstrs((v[i] == 0) >> (e.sum('*', i) == 0) for i in range(loc_len))
        m.addConstrs((v[i] == 0) >> (e.sum(i, '*') == 0) for i in range(loc_len))

        m.addConstrs((v[i] == 0) >> (d.sum('*', i) == 0) for i in range(loc_len))
        m.addConstrs(d.sum(h, '*') == 1 for h in range(home_len))

        # Optimize model
        # m.Params.lazyConstraints = 1
        m.optimize()
        for var in m.getVars():
            if var.x == 1:
                print('%s = %g' % (var.varName, var.x))
        print(e.prod(w).getValue())
        print('Obj: ', m.objVal)

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)
    # car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)
    #
    # basename, filename = os.path.split(input_file)
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    # output_file = utils.input_to_output(input_file, output_directory)
    #
    # convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
