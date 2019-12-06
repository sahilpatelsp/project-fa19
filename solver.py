import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import networkx as nx

from student_utils import *
from gurobipy import *
import gurobipy as gp
from collections import defaultdict
"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(input_file, list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    G = adjacency_matrix_to_graph(adjacency_matrix)[0]
    car_path_backtracking, drop_offs_backtracking = solve_backtracking(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[])
    did_Timeout, car_path_ilp, drop_offs_ilp = solve_ilp(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[])
    cost_backtracking = cost_of_solution(G, car_path_backtracking, drop_offs_backtracking)
    cost_ilp = cost_of_solution(G, car_path_ilp, drop_offs_ilp)
    if did_Timeout:
        f = open("time_limit.txt", "a")
        f.write(input_file + '\n')
        f.close()
    if cost_backtracking == "infinite":
        print("ILP")
        return car_path_ilp, drop_offs_ilp
    elif cost_ilp == "infinite":
        print("BACKTRACKING")
        return car_path_backtracking, drop_offs_backtracking
    elif cost_ilp <= cost_backtracking:
        print("ILP")
        return car_path_ilp, drop_offs_ilp
    else:
        print("BACKTRACKING")
        return car_path_backtracking, drop_offs_backtracking

def solve_backtracking(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
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
    home_indices = [list_of_locations.index(list_of_homes[i]) for i in range(len(list_of_homes))]
    G = adjacency_matrix_to_graph(adjacency_matrix)[0]
    spl = dict(nx.all_pairs_dijkstra_path_length(G))
    paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    path_weight = dict(nx.all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight'))
    loc_len = len(list_of_locations)
    home_len = len(list_of_homes)
    start_index = list_of_locations.index(starting_car_location)
    path_freq = {}
    for i in home_indices:
        path = paths[i][start_index]
        for j in range(len(path) - 1):
            curr_edge = (path[j], path[j + 1])
            if curr_edge in path_freq:
                path_freq[curr_edge] += 1
            else:
                path_freq[curr_edge] = 1
    #adj_mat = [[0 for _ in range(loc_len)] for i in range(loc_len)]
    adj_list = [0] * loc_len
    for i in range(loc_len):
        adj_list[i] = []
    for k in path_freq.keys():
        if path_freq[k] > 1:
            # print(k)
            adj_list[k[0]].append(k[1])
            adj_list[k[1]].append(k[0])
    path = printCircuit(adj_list,start_index)
    dropoffs = []
    [dropoffs.append(path_node) for path_node in path]
    dropoffs.append(start_index)
    for i in range(0, len(path)):
        if i == 0 or i == len(path) - 1:
            if path[i] in home_indices:
                dropoffs.append(path[i])
        else:
            if path[i] not in dropoffs:
                if path[i-1] == path[i+1] or path[i] in home_indices:
                    dropoffs.append(path[i])
    dropoff_indices = {}
    for dropoff in path:
        dropoff_indices[dropoff] = []
    for home in home_indices:
        minSoFar = (start_index, path_weight[start_index][home])
        for dropoff in dropoffs:
            path_cost = path_weight[dropoff][home]
            if path_cost  < minSoFar[1]:
                minSoFar = (dropoff, path_cost)
        dropoff_indices[minSoFar[0]].append(home)

    removal_indices = []
    for dropoff in dropoff_indices:
        if len(dropoff_indices[dropoff]) == 0:
            removal_indices.append(dropoff)
    [dropoff_indices.pop(dropoff) for dropoff in removal_indices]
    return path, dropoff_indices

def solve_ilp(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
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
        m = m.relax()
        #v variable for each location
        v = m.addVars(loc_len, vtype=GRB.BINARY, name="v")
        #e variable for each edge as a 1D array
        e = m.addVars(loc_len, loc_len, vtype=GRB.BINARY, name="e")
        u = m.addVars(loc_len, vtype=GRB.INTEGER, name="u")
        p = m.addVars(loc_len, loc_len, vtype=GRB.BINARY, name="p")
        d = m.addVars(home_len, loc_len, vtype=GRB.BINARY, name="d")
        m.update()

        m._loc_len = m.addVar(vtype=GRB.INTEGER)
        m._loc_len = loc_len
        m._e = e
        m._v = v
        m._d = d
        # Set objective
        m.setObjective(((2/3) * e.prod(w)) + d.prod(dist), GRB.MINIMIZE)

        m.addConstrs(e[i,j] == 0 for j in range(loc_len) for i in range(loc_len) if w[i,j] == 0)
        m.addConstr(v[start_index] == 1)
        m.addConstr(e.sum() >= v.sum())
        m.addConstrs((v[i] == 1) >> (e.sum('*', i) >= 1) for i in range(loc_len))
        m.addConstrs((v[i] == 1) >> (e.sum(i, '*') >= 1) for i in range(loc_len))
        m.addConstrs(e.sum('*', i) == e.sum(i, '*') for i in range(loc_len))
        m.addConstrs((v[i] == 0) >> (e.sum('*', i) == 0) for i in range(loc_len))
        m.addConstrs((v[i] == 0) >> (e.sum(i, '*') == 0) for i in range(loc_len))
        m.addConstrs((e[i,j] == 1) >> (v[i] + v[j] == 2) for i in range(loc_len) for j in range(loc_len) if i != j)

        m.addConstrs((v[i] == 0) >> (d.sum('*', i) == 0) for i in range(loc_len))
        m.addConstrs(d.sum(h, '*') == 1 for h in range(home_len))

        m.addConstr(u[start_index] == 1)
        m.addConstrs(u[i] >= 2 for i in range(loc_len) if i != start_index)
        m.addConstrs(u[i] <= loc_len - 1 for i in range(loc_len) if i != start_index)

        for i in range(loc_len):
            if i != start_index:
                for j in range(loc_len):
                    if j != start_index:
                        m.addConstr((u[i] - u[j] + 1) <= ((loc_len - 1) * (1 - e[i, j])))
        m.params.TimeLimit = 600
        m.optimize()
        # print('Obj: ', m.objVal)
        if m.Status == 9:
            did_Timeout = True
        else:
            did_Timeout = False
        # for var in m.getVars():
        #     if var.x > 0 and "p" not in var.varName:
        #         print('%s = %g' % (var.varName, var.x))
        # m.computeIIS()
        # m.write("infeasible.ilp")
        # m.write("file.lp")
        adj_list = [0] * loc_len
        for i in range(loc_len):
            adj_list[i] = []
        for i in range(loc_len):
            for j in range(loc_len):
                if m._e[i, j].x > 0.5:
                    adj_list[i].append(j)
        path = printCircuit(adj_list,start_index)
        dropoff_indices = {}
        for dropoff in path:
            dropoff_indices[dropoff] = []
        # d = m.addVars(home_len, loc_len, vtype=GRB.BINARY, name="d")
        for h in range(home_len):
            for i in range(loc_len):
                if m._d[h, i].x > 0.5:
                    dropoff_indices[i].append(list_of_locations.index(list_of_homes[h]))

        removal_indices = []
        for dropoff in dropoff_indices:
            if len(dropoff_indices[dropoff]) == 0:
                removal_indices.append(dropoff)
        [dropoff_indices.pop(dropoff) for dropoff in removal_indices]
        return did_Timeout, path, dropoff_indices

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
    car_path, drop_offs = solve(input_file, list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)

def printCircuit(adj, start_index):

    # adj represents the adjacency list of
    # the directed graph
    # edge_count represents the number of edges
    # emerging from a vertex
    edge_count = dict()

    for i in range(len(adj)):

        # find the count of edges to keep track
        # of unused edges
        edge_count[i] = len(adj[i])

    if len(adj) == 0:
        return # empty graph

    # Maintain a stack to keep vertices
    curr_path = []

    # vector to store final circuit
    circuit = []

    # start from any vertex
    curr_path.append(start_index)
    curr_v = start_index # Current vertex

    while len(curr_path):

        # If there's remaining edge
        if edge_count[curr_v]:

            # Push the vertex
            curr_path.append(curr_v)

            # Find the next vertex using an edge
            next_v = adj[curr_v][-1]

            # and remove that edge
            edge_count[curr_v] -= 1
            adj[curr_v].pop()

            # Move to next vertex
            curr_v = next_v

        # back-track to find remaining circuit
        else:
            circuit.append(curr_v)

            # Back-tracking
            curr_v = curr_path[-1]
            curr_path.pop()

    # we've got the circuit, now print it in reverse
    path = []
    for i in range(len(circuit) - 1, -1, -1):
        path.append(circuit[i])
    return path

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
