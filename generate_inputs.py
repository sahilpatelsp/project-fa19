import numpy as np
from faker import Faker
from collections import deque
fake = Faker()
num_locations = 200
num_homes = 100
locations = {}
homes = {}
root_name = ""
matrix = [['x' for _ in range(num_locations)] for i in range(num_locations)]
def generate_locations_and_homes():
    countries = []
    for i in range(num_locations):
        bool = False
        while not bool:
            country = fake.city()
            if (" " not in country) and (country not in countries) and (len(country) <= 20):
                countries.append(country)
                locations[i] = country
                bool = True
    chosen_numbers = []
    for i in range(num_homes):
        bool = False
        while not bool:
            number = np.random.randint(0, num_locations)
            if number not in chosen_numbers:
                chosen_numbers.append(number)
                homes[number] = locations[number]
                bool = True


generate_locations_and_homes()

def generate_input():
    #Step 1: Pick root
    current_locations = list(locations.keys())
    current_homes = list(homes.keys())
    not_homes = [item for item in current_locations if item not in current_homes]
    root_index = np.random.randint(len(not_homes))
    root = not_homes[root_index]
    root_name = locations[root]
    print(root_name)
    not_homes.remove(root)
    #Append and pop left
    queue = deque()
    queue.append(root)
    while not_homes:
        parent = queue.popleft()
        branching_factor = np.random.randint(2,4)
        for i in range(branching_factor):
            if not not_homes:
                break
            child = np.random.randint(len(not_homes))
            child = not_homes[child]
            not_homes.remove(child)
            edge_length = round(np.random.uniform(1000000000, 2000000000), 5)
            matrix[parent][child] = edge_length
            matrix[child][parent] = edge_length
            queue.append(child)
    while current_homes:
        parent = queue.popleft()
        branching_factor = np.random.randint(0,3)
        for i in range(branching_factor):
            if not current_homes:
                break
            child = np.random.randint(len(current_homes))
            child = current_homes[child]
            current_homes.remove(child)
            edge_length = round(np.random.uniform(1000000000, 2000000000), 5)
            matrix[parent][child] = edge_length
            matrix[child][parent] = edge_length
            queue.append(child)

def print_matrix():
    for i in range(num_locations):
        for j in range(num_locations):
            print(matrix[i][j],end =" ")
        print()

def print_list(lst):
    for i in range(len(lst)):
        print(lst[i], end=" ")
    print()
def printer():
    print(num_locations)
    print(num_homes)
    print_list(list(locations.values()))
    print_list(list(homes.values()))
    print_matrix()


generate_input()
printer()
