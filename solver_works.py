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
def subtourelim(m, where):
    if where == GRB.callback.MIPSOL:
        # print("HELLO")
        loc_len = m._loc_len
        v_sol = m.cbGetSolution([m._v[i] for i in range(loc_len)])
        v_1 = [i for i in range(loc_len) if v_sol[i] == 1]
        print("V")
        print(v_1)
        v1_len = len(v_1)

        v_in = {}
        for i in range(v1_len):
            v_in[i] = v_1[i]

        v_ni = {}
        for i in range(v1_len):
            v_ni[v_1[i]] = i

        #e_sol = m.cbGetSolution([m._e[i, j] for i in range(loc_len) for j in range(loc_len)])
        selected = []
        # make a list of edges selected in the solution
        for i in range(loc_len):
            sol = m.cbGetSolution([m._e[i,j] for j in range(loc_len)])
            selected += [(i,j) for j in range(loc_len) if sol[j] == 1]
        # find the shortest cycle in the selected edge list
        tour = subtour(selected, v_in, v_ni, v1_len, loc_len)
        print("TOUR")
        print(tour)
        if len(tour) < v1_len:
            # add a subtour elimination constraint
            expr = 0
            a = []
            for i in range(len(tour)):
                b = []
                for j in range(i+1, len(tour)):
                    expr += m._e[tour[i], tour[j]]
                    b.append((tour[i], tour[j]))
                a.append(b)
            print("EXPR")
            print(a)
            m.cbLazy(expr <= len(tour) - 1)


def subtour(edges, v_in, v_ni, v1_len, loc_len):
    print("EDGES")
    print(edges)
    visited = [False]*v1_len
    cycles = []
    lengths = []
    selected = [[] for i in range(v1_len)]
    for x,y in edges:
        selected[v_ni[x]].append(v_ni[y])
    print(selected)
    while True:
        current = visited.index(False)
        thiscycle = [v_in[current]]
        while True:
            visited[current] = True
            neighbors = [x for x in selected[current] if not visited[x]]
            if len(neighbors) == 0:
                break
            current = neighbors[0]
            thiscycle.append(v_in[current])

        print(thiscycle)
        cycles.append(thiscycle)
        lengths.append(len(thiscycle))
        if sum(lengths) == v1_len:
            break
    return cycles[lengths.index(min(lengths))]


# def subtourelim(m, where):
#     loc_len = m._loc_len
#     if where == GRB.callback.MIPSOL:
#         v_sol = m.cbGetSolution([m._v[i] for i in range(loc_len)])
#         v_1 = [i for i in range(loc_len) if v_sol[i] == 1]
#         e_sol = m.cbGetSolution([m._e[i, j] for i in range(loc_len) for j in range(loc_len)])
#
#         adj_M = [[1 if e_sol[i*loc_len + j] == 1 else 'x' for j in range(loc_len)] for i in range(loc_len)]
#         # for i in range(loc_len):
#         #     for j in range(loc_len):
#         #         if e_sol[i*loc_len + j] == 1:
#         #             adj_M[i][j] = 1
#
#         G = adjacency_matrix_to_graph(adj_M)[0]
#         v_s = list(nx.dfs_preorder_nodes(G, m._start_index))
#         if len(v_s) < len(v_1):
#             v_t = [el for el in v_1 if el not in v_s]
#             expr_in = 0
#             expr_out = 0
#             for i in v_s:
#                 for j in v_t:
#                     expr_in += m._e[i, j]
#                     expr_out += m._e[j, i]
#             m.cbLazy(expr_in >= 1)
#             m.cbLazy(expr_out >= 1)
#         # G = adjacency_matrix_to_graph(adj_M)[0]
#         # v_s = list(nx.dfs_preorder_nodes(G, m._start_index))
#         # while len(v_s) < len(v_1):
#         #     v_t = [el for el in v_1 if el not in v_s]
#         #     expr_in = 0
#         #     expr_out = 0
#         #     for i in v_s:
#         #         for j in v_t:
#         #             expr_in += m._e[i, j]
#         #             expr_out += m._e[j, i]
#         #     m.cbLazy(expr_in >= 1)
#         #     m.cbLazy(expr_out >= 1)
#         #     v_s += list(nx.dfs_preorder_nodes(G, v_t[0]))

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
        m = m.relax()
        #v variable for each location
        v = m.addVars(loc_len, vtype=GRB.BINARY, name="v")
        #e variable for each edge as a 1D array
        e = m.addVars(loc_len, loc_len, vtype=GRB.BINARY, name="e")

        # e = {}
        # for i in range(loc_len):
        #     for j in range(i+1):
        #         e[i,j] = m.addVar(vtype=GRB.BINARY, name="e")
        #         e[j,i] = e[i,j]

        u = m.addVars(loc_len, vtype=GRB.INTEGER, name="u")
        p = m.addVars(loc_len, loc_len, vtype=GRB.BINARY, name="p")
        d = m.addVars(home_len, loc_len, vtype=GRB.BINARY, name="d")
        m.update()
        # m._start_index = m.addVar(vtype=GRB.INTEGER)
        # m._start_index= start_index
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

        # m.addConstrs((e[i,j] == 1) >> (e[j,i] == 0) for i in range(loc_len) for j in range(loc_len))

        # x = m.addVar(vtype=GRB.INTEGER, name="x")
        # m.addConstr(x >= 1)
        m.addConstrs((v[i] == 1) >> (e.sum('*', i) >= 1) for i in range(loc_len))
        m.addConstrs((v[i] == 1) >> (e.sum(i, '*') >= 1) for i in range(loc_len))
        m.addConstrs(e.sum('*', i) == e.sum(i, '*') for i in range(loc_len))
        m.addConstrs((v[i] == 0) >> (e.sum('*', i) == 0) for i in range(loc_len))
        m.addConstrs((v[i] == 0) >> (e.sum(i, '*') == 0) for i in range(loc_len))
        m.addConstrs((e[i,j] == 1) >> (v[i] + v[j] == 2) for i in range(loc_len) for j in range(loc_len) if i != j)
        # m.addConstrs((e[i,j] == 0) >> (v[i] + v[j] == 0) for i in range(loc_len) for j in range(loc_len) if i != j)


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

        # m.params.LazyConstraints = 1
        m.params.TimeLimit = 180
        m.optimize()
        for var in m.getVars():
            if var.x > 0 and "p" not in var.varName:
                print('%s = %g' % (var.varName, var.x))
        getOutputValues2(m._v, m._e, m._d, w, loc_len, start_index)
        print('Obj: ', m.objVal)
        # m.computeIIS()
        # m.write("infeasible.ilp")
        # m.write("file.lp")

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

def getOutputValues2(vertices, edges, dropoffs, weight, loc_len, start_index):
    adj_matrix = [[1 if edges[i, j].x == 1 else 'x' for j in range(loc_len)] for i in range(loc_len)]
    G = adjacency_matrix_to_graph(adj_matrix)[0]
    v_s = list(nx.dfs_preorder_nodes(G, start_index))
    print(v_s)
def getOutputValues(vertices, edges, dropoffs, weight, loc_len, start_index):
    actual_vertices = [i for i in range(loc_len) if vertices[i].x == 1]
    # new_start_index = actual_vertices.index(start_index)
    # g = Graph(len(actual_vertices))
    # [g.addEdge(i,j) for j in range(len(actual_vertices)) for i in range(len(actual_vertices)) if edges[actual_vertices[i], actual_vertices[j]].x == 1]
    # print(g.isEulerianCycle())
    #
    adj = [0] * len(actual_vertices)
    for i in range(len(actual_vertices)):
        adj[i] = []
    # [adj[i].append(j) for j in range(len(actual_vertices)) for i in range(len(actual_vertices)) if edges[actual_vertices[i], actual_vertices[j]].x == 1]
    # print(printCircuit(adj, actual_vertices, new_start_index))
    new_start_index = actual_vertices.index(start_index)
    adj_matrix = [[weight[actual_vertices[i], actual_vertices[j]] if edges[actual_vertices[i], actual_vertices[j]].x == 1 else 'x' for j in range(len(actual_vertices))] for i in range(len(actual_vertices))]
    # for i in range(len(actual_vertices)):
    #     for j in range(len(actual_vertices)):
    #         print(adj_matrix[i][j], end = ' ')
    #     print()
    G = adjacency_matrix_to_graph(adj_matrix)[0]
    circuit = list(nx.eulerian_circuit(G,source=new_start_index))
    new_circuit = [(actual_vertices[i], actual_vertices[j]) for i,j in circuit]
    print(new_circuit)

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
