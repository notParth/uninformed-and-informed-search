# Implementation of {Project one: maze} by Parth Patel and Mustafa Sadiq

import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math
from queue import PriorityQueue

# type of maze cells:
# 0) blocked("#") 
# 1) open (" ")  
# 2) traversed (*) 
# 3) Start ("S") 
# 4) Goal ("G") 

# make_maze creates a maze of dimensions (dim * dim) and blocked cell probability of (p)
def make_maze(dim, p):
    maze = np.empty([dim, dim], dtype = str)
    for x in range(dim):
        for y in range(dim):
            if(rand.random() <= p):
                maze[x][y] = "X"
            else:
                maze[x][y] = " "
    maze[0][0] = "S"
    maze[dim - 1][dim - 1] = "G"
    return maze

# prints a given maze
def print_maze(maze):
    print('-'*(len(maze)*4), end="")
    print()
    for x in maze:
        for y in x:
            print(f'| {y} ', end="")
        print("|", end="")
        print()
        print('-'*(len(maze)*4), end="")
        print()


# prints the path found by the search on the maze
def print_path(maze, path):
    new_maze = maze
    while path.parent is not None:
        parent = path.parent.coordinates
        maze[parent[0]][parent[1]] = "*"
        path = path.parent
    new_maze[0][0] = "S"
    print_maze(new_maze)

# removes path from matrix
def remove_path(maze, path):
    while path.parent is not None:
        parent = path.parent.coordinates
        maze[parent[0]][parent[1]] = " "
        path = path.parent
    maze[0][0] = "S"

# this function returns the valid children of a given position on the maze
# valid children are the ones that are not blocked("X") and are not outside the maze
def generate_children(maze, position):
    x = position[0]
    y = position[1]
    dim = len(maze)
    valid_children = []
    # check up, down, left and right of the position for valid children
    if x+1 <= dim - 1 and maze[x+1][y] != "X": 
        valid_children.append(Node((x+1, y)))
    if x-1 >= 0 and maze[x-1][y] != "X": 
        valid_children.append(Node((x-1, y)))
    if y+1 <= dim - 1 and maze[x][y+1] != "X": 
        valid_children.append(Node((x, y+1)))
    if y-1 >= 0 and maze[x][y-1] != "X": 
        valid_children.append(Node((x, y-1)))
    
    return valid_children

# Used to construct the graph
class Node:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.parent = None
        self.level = 0
    def __lt__(self, other):
        return

# Calculate euclidean distance
def euclid_dist(start, goal):
    return math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)

# Depth first search
def dfs(maze, start, goal):
    fringe = []
    fringe.append(Node(start))
    closed_set = {''}

    while fringe:
        current_state = fringe.pop()
        if current_state.coordinates == goal:
            return current_state
        for x in generate_children(maze, current_state.coordinates):
            if(x.coordinates not in closed_set):
                x.parent = current_state
                fringe.append(x)
        closed_set.add(current_state.coordinates)
    
    return None

# Breadth First Search
def bfs(maze, start, goal):
    fringe = []
    fringe.append(Node(start))
    closed_set = {''}

    while fringe:
        current_state = fringe.pop(0)
        if current_state.coordinates == goal:
            return current_state
        for x in generate_children(maze, current_state.coordinates):
            if(x.coordinates not in closed_set):
                x.parent = current_state
                fringe.append(x)
        closed_set.add(current_state.coordinates)
    
    return None

# A*
def a_star(maze, start, goal):
    fringe = PriorityQueue()
    this_node = Node(start)
    this_node.level = 0
    fringe.put(((this_node.level+euclid_dist(start, goal)), this_node))

    while not fringe.empty():
        current_state = fringe.get()[1]
        if current_state.coordinates == goal:
            return current_state
        for x in generate_children(maze, current_state.coordinates):
                x.parent = current_state
                x.level = current_state.level + 1
                coordinates = x.coordinates
                level = x.level
                fringe.put((level+euclid_dist(coordinates, goal),x))
    return None

this_maze = make_maze(10, 0.3)
print("Randomly generated maze:")
print_maze(this_maze)

# solving this_maze using dfs
answer_dfs = dfs(this_maze, (0,0), (9,9))
if answer_dfs != None:
    print("Path found by DFS")
    print_path(this_maze, answer_dfs)
    remove_path(this_maze, answer_dfs)
    # solving this_maze using bfs
    answer_bfs = bfs(this_maze, (0,0), (9,9))
    print("Path found by BFS:")
    print_path(this_maze, answer_bfs)
    remove_path(this_maze, answer_bfs)
    #solving this_maze using a*
    answer_a_star = a_star(this_maze, (0,0), (9,9))
    print("Path found by A*:")
    print_path(this_maze, answer_a_star)
    remove_path(this_maze, answer_a_star)
else:
    print("no solution")

# graph generation for dfs: ‘obstacle density p’ vs 
# ‘probability that S can be reached from G’.
x_range = np.arange(0.0, 1.0, 0.01)
y_range = np.empty(100)
cnt = 0
dim = 50
tries_per_density = 100
for i in x_range:
    success = 0
    for j in range (tries_per_density):
        this_maze = make_maze(dim, i)
        answer = dfs(this_maze, (0,0), (dim-1, dim-1))
        if answer != None:
            success += 1
    y_range[cnt] = success / tries_per_density
    cnt += 1


plt.plot(x_range, y_range)
plt.title("Probability that S can be reached from G for DFS vs Obstacle density p")
plt.xlabel("obstacle density p")
plt.ylabel("probability that S can be reached from G")
plt.show()