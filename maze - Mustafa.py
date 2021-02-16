import random as rand
import numpy as np
from collections import deque
import heapq
from math import sqrt
import matplotlib.pyplot as plt
import time
 
####################### Make maze ##############################
def make_maze(dim, p):
    maze = np.empty([dim, dim], dtype = str)
    for x in range(dim):
        for y in range(dim):
            if(rand.random() <= p):
                maze[x][y] = 'X'
            else:
                maze[x][y] = ' ' 
    maze[0][0] = 'S'
    maze[dim - 1][dim - 1] = 'G'
    return maze

################### Print ascii maze ##############################
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

################# Take n steps on maze with given path #################
def take_n_steps(maze, path, steps=None):
    if steps == None:
        for point in path:
            maze[point[0]][point[1]] = '*'
    elif steps == -1:
        for point in path:
            maze[point[0]][point[1]] = ' '
    else:
        for point in path[:steps]:
            maze[point[0]][point[1]] = '*'

##################### Get neighbours of a given point in a maze ####################
def get_neighbours(maze, current):
    x = current[0]
    y = current[1]
    dim = len(maze)
    neighbours = []

    if x-1 >= 0 and maze[x-1][y] != 'X': 
        neighbours.append((x-1, y))
    if y-1 >= 0 and maze[x][y-1] != 'X': 
        neighbours.append((x, y-1))
    if y+1 <= dim - 1 and maze[x][y+1] != 'X': 
        neighbours.append((x, y+1))
    if x+1 <= dim - 1 and maze[x+1][y] != 'X': 
        neighbours.append((x+1, y))   
    
    return neighbours

def get_nonfire_neighbours(maze, current):
    x = current[0]
    y = current[1]
    dim = len(maze)
    neighbours = []

    if x-1 >= 0 and maze[x-1][y] != 'X' and maze[x-1][y] != 'F': 
        neighbours.append((x-1, y))
    if y-1 >= 0 and maze[x][y-1] != 'X' and maze[x][y-1] != 'F':
        neighbours.append((x, y-1))
    if y+1 <= dim - 1 and maze[x][y+1] != 'X' and maze[x][y+1] != 'F':
        neighbours.append((x, y+1))
    if x+1 <= dim - 1 and maze[x+1][y] != 'X' and maze[x+1][y] != 'F':
        neighbours.append((x+1, y))   
    
    return neighbours

########### test neighbour #############
# maze = make_maze(10,0.1)
# maze[1][2] = 'F'
# print(get_nonfire_neighbours(maze, (2,2)))



###################### DFS = Find a path given a maze, start, goal ###########################
def dfs(maze, start, goal):
    fringe = [start] 
    tree = dict()
    tree[start] = None

    while  fringe:
        current = fringe.pop()
        if current == goal:
            current = goal
            path = []
            while current != start:
                path.append(current)
                current = tree[current]
            path.append(start)
            path.reverse()
            return path
        for neighbour in get_nonfire_neighbours(maze, current):
            if neighbour not in tree:
                fringe.append(neighbour)
                tree[neighbour] = current

    return None

###################### BFS = Find a path given a maze, start, goal ###########################
def bfs(maze, start, goal):
    fringe = deque() 
    fringe.append(start)
    tree = dict()
    tree[start] = None

    while  fringe:
        current = fringe.popleft()
        if current == goal:
            current = goal
            path = []
            while current != start:
                path.append(current)
                current = tree[current]
            path.append(start)
            path.reverse()
            return path
        for neighbour in get_nonfire_neighbours(maze, current):
            if neighbour not in tree:
                fringe.append(neighbour)
                tree[neighbour] = current

    return None

#################### Euclidean distance #######################################
def distance(start, end):
    return sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)

###################### A* = Find a path given a maze, start, goal ###########################
def astar(maze, start, goal):
    fringe = [] 
    heapq.heappush(fringe, (0,start))
    tree = dict()
    cost_tree = dict()
    tree[start] = None
    cost_tree[start] = 0

    while  fringe:
        current = heapq.heappop(fringe)[1]
        if current == goal:
            current = goal
            path = []
            while current != start:
                path.append(current)
                current = tree[current]
            path.append(start)
            path.reverse()
            return path
        for neighbour in get_nonfire_neighbours(maze, current):
            new_cost = cost_tree[current] + 1
            if neighbour not in cost_tree or new_cost < cost_tree[neighbour]:
                cost_tree[neighbour] = new_cost
                priority = new_cost + distance(goal, neighbour)
                heapq.heappush(fringe, (priority, neighbour))
                tree[neighbour] = current

    return None

###################################################################################
############################### Problem 1-4 solving area ##############################
##################################################################################

########################## Test all three searches using one maze #################
# dim = 10
# p = 0.3
# maze = make_maze(dim,p)
# print_maze(maze)
# print("\n\nDepth first search: ")

# path = dfs(maze, (0,0), (dim-1,dim-1))
# take_n_steps(maze, path)
# print_maze(maze)
# print("\n\nBreadth first search:")

# take_n_steps(maze, path, -1)

# path = bfs(maze, (0,0), (dim-1,dim-1))
# take_n_steps(maze, path)
# print_maze(maze)
# print("\n\nA* search:")
    
# take_n_steps(maze, path, -1)

# path = astar(maze, (0,0), (dim-1,dim-1))
# take_n_steps(maze, path)
# print_maze(maze)
# print("\n\n")
    
############################### Question 2 ########################################## 
# For as large a dimension as your system can handle, generate a plot of `obstacle density p' vs `probability that S can be reached from G'.

# dim = 100
# dataX = []
# dataY = []
# runs = 100

# for density in np.linspace(0,1,100):
#     success = 0
#     for run in range(runs):
#         maze = make_maze(dim, density)
#         if dfs(maze, (0,0), (dim-1,dim-1)) != None:
#             success += 1
        
#     dataX.append(success/runs)
#     dataY.append(density)

# plt.plot(dataX, dataY)
# plt.title("`obstacle density p' vs `probability that S can be reached from G'")
# plt.ylabel("obstacle density p")
# plt.xlabel("probability that S can be reached from G")
# plt.show()

    
############################### Question 3 ########################################## 
# Write BFS and A* algorithms (using the euclidean distance metric) that take a maze and determine the shortest path from S to G if one exists.
#  For as large a dimension as your system can handle, generate a plot of the average `number of nodes explored by 
# BFS - number of nodes explored by A*' vs `obstacle density p'. If there is no path from S to G, what should this difference be?

# def bfs_counter(maze, start, goal):
#     fringe = deque() 
#     fringe.append(start)
#     tree = dict()
#     tree[start] = None
#     count = 0

#     while  fringe:
#         current = fringe.popleft()
#         count += 1
#         if current == goal:
#             return count
#         for neighbour in get_neighbours(maze, current):
#             if neighbour not in tree:
#                 fringe.append(neighbour)
#                 tree[neighbour] = current
                

#     return count

# def astar_counter(maze, start, goal):
#     fringe = [] 
#     heapq.heappush(fringe, (0,start))
#     tree = dict()
#     cost_tree = dict()
#     tree[start] = None
#     cost_tree[start] = 0
#     count = 0

#     while  fringe:
#         current = heapq.heappop(fringe)[1]
#         count += 1
#         if current == goal:
#             return count
#         for neighbour in get_neighbours(maze, current):
#             new_cost = cost_tree[current] + 1
#             if neighbour not in cost_tree or new_cost < cost_tree[neighbour]:
#                 cost_tree[neighbour] = new_cost
#                 priority = new_cost + distance(goal, neighbour)
#                 heapq.heappush(fringe, (priority, neighbour))                
#                 tree[neighbour] = current

#     return count

# dim = 100
# dataX = []
# dataY = []
# runs = 100

# for density in np.linspace(0,1,100):
#     difference = 0
#     for run in range(runs):
#         maze = make_maze(dim, density)
#         bfs_nodes_explored = bfs_counter(maze, (0,0), (dim-1,dim-1))  
#         astar_nodes_explored = astar_counter(maze, (0,0), (dim-1,dim-1))
#         difference += bfs_nodes_explored - astar_nodes_explored
        
#     dataX.append(density)
#     dataY.append(difference/runs)

# plt.plot(dataX, dataY)
# plt.title("`number of nodes explored by BFS - number of nodes explored by A*' vs `obstacle density p'")
# plt.ylabel("number of nodes explored by BFS - number of nodes explored by A*")
# plt.xlabel("obstacle density p")
# plt.show()

########################################### Problem 4 ###################################
# What's the largest dimension you can solve using DFS at p = 0:3 in less than a minute?
# What's the largest dimension you can solve using BFS at p = 0:3 in less than a minute? 
# What's the largest dimension you can solve using A at p = 0:3 in less than a minute?


# dim = 10500
# runs = 5
# difference = 0

# for run in range(runs):
#     while True:
#         maze = make_maze(dim, 0.3)
#         starting_time = time.time()
#         # change search type here
#         path = dfs(maze, (0,0), (dim-1, dim-1))
#         difference = time.time()-starting_time
#         if path != None:
#             print("Path found in",difference,"seconds with",dim,"dim")
#             if difference > 60:
#                 break
#             dim += 100
#         else:
#             print("Path not found in",difference,"seconds with",dim,"dim")

# print("Largest dimension found:", dim)   


###########################################################################
################################# PART 2 ##################################
###########################################################################

def advance_fire_one_step(maze, q=0.3):
    copy = np.copy(maze)
    dim = len(maze)
    for x in range(dim):
        for y in range(dim):
            if maze[x][y] != 'F' and maze[x][y] != 'X':
                k = 0
                for neighbour in get_neighbours(maze, (x, y)):
                    if maze[neighbour[0]][neighbour[1]] == 'F':
                        k += 1
                    prob = 1 - (1-q)**k
                    if rand.random() <= prob:
                        copy[x][y] = 'F'

    return copy

################## testing advance_fire_one_step ####################################
# maze = make_maze(10,0.1)
# print_maze(maze)
# maze[1][2] = 'F'
# maze[1][1] = 'X'
# maze = advance_fire_one_step(maze, 1.0)
# print_maze(maze)


def take_nth_step_with_fire(maze, path, step=0):
    if maze[path[step][0]][path[step][1]] != 'F':
        maze[path[step][0]][path[step][1]] = '*'
        return True
    else:
        return False  


###################################################################################
############################### Problem 5-8 solving area ##########################
################################################################################## 




################################## Strategy 1 ############################################


# dim = 10
# maze = make_maze(dim, 0.3)
# start = (0,0)
# fire = (0, dim-1)

# if dfs(maze, start, fire) != None:
#     maze[fire[0]][fire[1]] = 'F'
#     print_maze(maze)
#     path = dfs(maze, (0,0), (dim-1, dim-1))

#     if (path != None):
#         for step in range(len(path)):
#             if take_nth_step_with_fire(maze, path, step):
#                 print("\n\n")
#                 print_maze(maze)
#                 maze = advance_fire_one_step(maze, 1.0)

                
#                 if path[step] == (dim-1,dim-1):
#                     print("path found.")
#             else:
#                 print("stepped into fire!")
#                 break    
#     else:
#         print("No path found.")
# else:

#     maze[fire[0]][fire[1]] = 'F'
#     print_maze(maze)
#     print("Cant reach fire.")


################################## Strategy 2 ############################################


# dim = 10
# maze = make_maze(dim, 0.3)
# start = (0,0)
# goal = (dim-1, dim-1)
# fire = (0, dim-1)
# maze[fire[0]][fire[1]] = ' '

# if dfs(maze, start, fire) != None:
#     maze[fire[0]][fire[1]] = 'F'
#     print_maze(maze)
#     while(True):           
#         path = dfs(maze, start, goal)        
#         if path != None:                  
#             take_n_steps(maze, path, 1)       
#             print_maze(maze)  
#             maze = advance_fire_one_step(maze, 1.0)                 
#             start = path[1]              
#             if start == goal:
#                 print("Path found")
#                 break
#         else:
#             print("\n\nNo path found.")
#             print_maze(maze)
#             break
# else:    
#     maze[fire[0]][fire[1]] = 'F'
#     print_maze(maze)
#     print("\n\nCant reach fire.")

################################## Strategy 3 ############################################


################################## Problem 6, Strategy 1 ############################################

dim = 20
runs = 25
p = 0.3
start = (0,0)
goal = (dim-1, dim-1)

dataX = []
dataY = []

for q in np.linspace(0,1,100):
    success = 0
    kept_runs = 0
    for run in range(runs):
        # print(run)
        maze = make_maze(dim, p)
        fire_x = rand.randrange(1,dim-1)
        fire_y = rand.randrange(1,dim-1)
        maze[fire_x][fire_y] = ' '
        if bfs(maze, start, (fire_x, fire_y)) != None:
            maze[fire_x][fire_y] = 'F'
            path = bfs(maze, start, goal)
            if path != None:
                kept_runs += 1
                for step in range(len(path)):
                    if take_nth_step_with_fire(maze, path, step):
                        maze = advance_fire_one_step(maze, q)
                        if path[step] == goal:
                            success += 1
                            break
    print("At q:",q,"runs kept",kept_runs)
    dataX.append(q)
    dataY.append(success/kept_runs)

plt.plot(dataX, dataY)
plt.title("Strategy 1 (`average strategy success rate' vs `flammability q')")
plt.ylabel("average strategy success rate")
plt.xlabel("flammability q")
plt.show()

################################## Problem 6, Strategy 2 ############################################

# dim = 30
# runs = 20
# p = 0.3

# goal = (dim-1, dim-1)

# dataX = []
# dataY = []

# for q in np.linspace(0,1,100):
#     success = 0
#     kept_runs = 0
#     for run in range(runs):
#         print(run)
#         maze = make_maze(dim, p)
#         fire_x = rand.randrange(1,dim-1)
#         fire_y = rand.randrange(1,dim-1)
#         maze[fire_x][fire_y] = ' '
#         start = (0,0)
#         if bfs(maze, start, (fire_x, fire_y)) != None:            
#             maze[fire_x][fire_y] = 'F'
#             path = bfs(maze, start, goal)
#             if path != None:
#                 kept_runs += 1
#                 while (True):
#                     if path != None:
#                         take_n_steps(maze, path, 1)
#                         maze = advance_fire_one_step(maze, q)
#                         start = path[1]
#                         if start == goal:
#                             success += 1
#                             break
#                     else:
#                         break
#                     path = bfs(maze, start, goal)

#     print("At q:",q,"runs kept",kept_runs)
#     dataX.append(q)
#     dataY.append(success/kept_runs)

# plt.plot(dataX, dataY)
# plt.title("Strategy 2 (`average strategy success rate' vs `flammability q')")
# plt.ylabel("average strategy success rate")
# plt.xlabel("flammability q")
# plt.show()

################################## Problem 6, Strategy 3 ############################################

