# Implementation of {Project one: maze} by Parth Patel and Mustafa Sadiq

import numpy as np
import random as rand

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

#prints a given maze
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


#prints the path found by the search on the maze
def print_path(maze, path):
    new_maze = maze
    while path.parent is not None:
        parent = path.parent.coordinates
        maze[parent[0]][parent[1]] = "*"
        path = path.parent
    print_maze(new_maze)

# this function returns the valid children of a given position on the maze
# valid children are the ones that are not blocked("#") and are not outside the maze
def generate_children(maze, position):
    x = position[0]
    y = position[1]
    dim = len(maze)
    valid_children = []
    #check up, down, left and right of the position for valid children
    if x+1 <= dim - 1 and maze[x+1][y] != "X": 
        valid_children.append(Node((x+1, y)))
    if x-1 >= 0 and maze[x-1][y] != "X": 
        valid_children.append(Node((x-1, y)))
    if y+1 <= dim - 1 and maze[x][y+1] != "X": 
        valid_children.append(Node((x, y+1)))
    if y-1 >= 0 and maze[x][y-1] != "X": 
        valid_children.append(Node((x, y-1)))
    
    return valid_children

#Used to construct the graph
class Node:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.parent = None

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

dim = 10
this_maze = make_maze(dim, 0.3)
print_maze(this_maze)
answer = dfs(this_maze, (0,0), (dim-1, dim-1))
if answer is None:
    print("No solution")
else:
    print("success")
    print_path(this_maze, answer)
