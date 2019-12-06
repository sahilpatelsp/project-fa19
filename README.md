# CS 170 Fall 2019 Project
#### By Solomon Joseph, Sahil Patel, and Franklin Chian

##Prerequisites
1. Python: We used version 3.6.2, but any version of python3 <= 3.7 should work for Gurobi purposes. Please visit https://www.python.org/downloads/ if you do not currently have python installed in your machine.
2. Python package manager: We used pip3, but any python package manager should do the trick. With help installing pip3, please consult https://pip.pypa.io/en/stable/
3. Networkx (https://networkx.github.io/)
4. Gurobi Optimizer

##Installation Guide

###Installing Gurobi
Our solution uses Gurobi Optimizer 9.0 under an academic license to formulate and solve ILPs for each graph. For the following steps, please ensure that you are connected to an academic network.

1. Register for a free Gurobi academic license at https://www.gurobi.com/downloads/end-user-license-agreement-academic/
2. Download Gurobi optimizer from https://www.gurobi.com/downloads/
3. Once installed, run ‘grbgetkey’ using the argument provided (ex: grbgetkey ae36ac20-16e6-acd2-f242-4da6e765fa0a) to store the license on your local machine
4. Navigate into the gurobi installation path
* For mac, this defaults to /Library/gurobi_server811/mac64
* For windows, this defaults to c:/gurobi_server811/win64
5. Run `python3 setup.py` to add gurobi to your python libraries

If you are facing any other problems, please refer to https://www.gurobi.com/documentation/8.1/remoteservices/installation.html

###Installing networkx
To install networkx, simply run the following command:
`pip3 install networkx`

###Running the solver
1. Navigate to the location of this project
2. Run python3 solver.py --all ./inputs ./outputs/

###Solver.py functions
1. solve(): Computes the cost of a path using both backtracking and ilp, and returns the minimum of both.
2. solve_backtracking(): Creates a path using a reverse  all-pairs shortest path from homes to the start index.
3. solve_ilp(): Creates a path using an integer linear programming formulation.
4. printCircuit(): Conducts Hierholzer's algorithm for directed graphs to find a eulerian circuit given edges in the graph and a start vertex. Code from https://www.geeksforgeeks.org/hierholzers-algorithm-directed-graph/
