import numpy as np
import random
from os import system
from time import sleep
import queue
import collections
import time

# Initialize grid size (mxn)
m, n = 10, 10
# Initialize Obstacle
num_obstacles = 10
# Initialize Beam Width 
beta = 3
# Control the speed of the snake
delay = 0.1
# Intialize Tabu list length
t = 30
# Declare the list required for Tabu search
de = collections.deque([], maxlen=t)
# Declare the grid as mxn numpy array
grid = np.zeros((m, n), dtype=np.int8)

# Input Algorithm and Heuristic Choice
algo_choice = int(input("Choose Algorithm \n0 for BreadthFirstSearch \n1 for BestFirstSearch \n2 for HillClimbingSearch \n3 for VariableNeighbourhoodDescent \n4 for BeamSearch \n5 for TabuSearch: "))
heuristic_choice = int(input("Heuristic(1 or 2 or 3): "))


# Generate food in the empty spaces of grid
def randomFoodGeneration(snake, obstacles, prev_food):
	while True:
		x, y = random.randrange(m), random.randrange(n)
		if [x,y] not in snake and [x,y] not in obstacles and [x,y] != prev_food:
			food = [x, y]
			break
	return food


# Declare and Initialize Snake Position
def initialSnake(m, n):
	snake = list()
	snake.append([int(m/2), int(n/2)])
	snake.append([int(m/2), int(n/2)-1])
	snake.append([int(m/2), int(n/2)-2])
	return snake


# Generate the grid given snake, food, and obstacles indices
def makeGrid(snake, food, obstacles):
	grid = np.zeros((m, n), dtype=np.int8)
	grid[food[0], food[1]] = 2
	for x, y in snake:
		grid[x, y] = 1
	for x, y in obstacles:
		grid[x, y] = 3
	return grid


# Place Obstacles Randomly in the empty spaces in grid
def plantObstacles(grid, num):
	obstacles = list()
	for i in range(num):
		while True:
			x, y = random.randrange(0, m), random.randrange(0, n)
			if grid[x, y] == 0:
				obstacles.append([x, y])
				grid[x, y] = 3
				break
	return obstacles


# Print the grid:
	# Snake: O for head and 0 for body
	# Food: F
	# Obstacles: X
def printGrid(grid, snake, obstacles, food):
	m, n = np.shape(grid)
	print("*", end="")
	for i in range(n):
		print("--", end="")
	print("*")
	for i in range(m):
		print("|", end="")
		for j in range(n):
			if [i,j] == snake[0]:
				print("O", end=" ")
			elif [i,j] in snake[1:]:
				print("0", end=" ")
			elif [i,j] in obstacles:
				print("X", end=" ")
			elif [i,j] == food:
				print("F", end=" ")
			else:
				print(" ", end=" ")
		print("|")
	print("*", end="")
	for i in range(n):
		print("--", end="")
	print("*")

	return


# Given current, goal state and heuristic choice assigns 
# and outputs the value of the given state
def heuristicVal(curr, goal, heuristic):
	if heuristic == 1:
		return abs(curr[0]-goal[0]) + abs(curr[1]-goal[1])
	elif heuristic == 2:
		return (curr[0]-goal[0])**2 + (curr[1]-goal[1])**2
	else:
		return 1000 - (abs(curr[0]-goal[0]) + abs(curr[1]-goal[1]))


# 0:Breadth First Search
def BFS(graph, start, dest, pred, visited):
	adj = [start]
	visited[start[0], start[1]] = 1
	
	while len(adj) != 0:
		x = adj.pop(0)
		for i in MoveGen(graph, x[0], x[1]):
			if visited[i[0], i[1]]==0:
				adj.append(i)
				visited[i[0], i[1]] = 1
				pred[i[0]][i[1]] = x
			
			if GoalTest(i, dest):
				return True, pred
	return False, pred


# 1:Best First Search
def BestFirstSearch(graph, start, dest, pred, visited, heuristic):
	adj = queue.PriorityQueue()
	adj.put([heuristicVal(start, dest, heuristic), start])
	explored = 0
	
	visited[start[0], start[1]] = 1
	
	while adj.qsize() != 0:
		h, x = adj.get()
		for i in MoveGen(graph, x[0], x[1]):
			if visited[i[0], i[1]]==0:
				explored += 1
				adj.put([heuristicVal(i, dest, heuristic), i])
				visited[i[0], i[1]] = 1
				pred[i[0]][i[1]] = x
			
			if GoalTest(i, dest):
				return True, pred, explored
	return False, pred, explored


# 2:Hill Climbing Search
def HillClimbingSearch(graph, start, dest, heuristic):
	neighbours = MoveGen(graph, start[0], start[1])
	if len(neighbours)==0:
		exitGame("No Possible neighbours to go to!")
	minh, minn = heuristicVal(start, dest, heuristic), start
	flag = 0
	for neigh in neighbours:
		if heuristicVal(neigh, dest, heuristic) < minh:
			minh = heuristicVal(neigh, dest, heuristic)
			minn = neigh
			flag = 1
	if flag == 0:
		return False, None
	else:
		return True, minn


# Breadth First Search upto a given depth
# Helper function for Variable Neighbourhood Descent Search
def BFS_limit(grid, start, dest, heuristic_choice, limit):
	visited = np.zeros((np.shape(grid)[0], np.shape(grid)[1]), dtype=np.int8)
	x = list()
	[x.append([0, 0]) for i in range(n)]
	pred = [x.copy() for i in range(m)]

	level = 0
	adj = [[start, level]]
	visited[start[0], start[1]] = 1
	
	currHeuristic = heuristicVal(start, dest, heuristic_choice)

	while len(adj) != 0:
		x, currLevel = adj.pop(0)
		if currLevel > limit:
			return False, pred, visited, None
		for i in MoveGen(grid, x[0], x[1]):
			if visited[i[0], i[1]]==0:
				adj.append([i, currLevel + 1])
				visited[i[0], i[1]] = 1
				pred[i[0]][i[1]] = x
			
		if currHeuristic > heuristicVal(i, dest, heuristic_choice):
			return True, pred, visited, i
	return False, pred, visited, None


# 3:Variable Neighbourhood Descent Search
def VariableNeighbourhoodDescent(grid, start, dest, heuristic_choice):	
	m, n = np.shape(grid)
	for i in range(m*n):
		result, pred, visited, pseudoDest = BFS_limit(grid, start, dest, heuristic_choice, i)
		if result == True:
			return True, pred, i, pseudoDest
	return False, pred, i, pseudoDest


# Given queue of Beam Search returns best beam width number of states according to the heuristic function
# Helper function for Beam Search
def Refine(adj, n, dest, heuristic, beamWidth):
	y = list()
	for ele in adj:
		y.append([ele[1], heuristicVal(ele[0], dest, heuristic), ele[0]])
	y = sorted(y, key = lambda x: (x[0], x[1]))
	
	l = 0
	for i in range(len(y)):
		if y[i][0] == n+1:
			l = i
			break
	y = y[:l+beamWidth]

	x = list()
	for i in y:
		x.append([i[2], i[0]])
	return x


# 4:Beam Search
def BeamSearch(graph, start, dest, pred, visited, heuristic, beta):
	currLevel = 0
	adj = [[start, currLevel]]
	visited[start[0], start[1]] = 1
	explored = 0
	while len(adj) != 0:
		adj = Refine(adj, currLevel, dest, heuristic, beta)
		x, currLevel = adj.pop(0)
		for i in MoveGen(graph, x[0], x[1]):
			if visited[i[0], i[1]]==0:
				explored += 1
				adj.append([i, currLevel+1])
				visited[i[0], i[1]] = 1
				pred[i[0]][i[1]] = x
			
			if GoalTest(i, dest):
				return True, pred, explored
	return False, pred, explored


# 5:Tabu Search
def TabuSearch(graph, start, dest, heuristic):
	global de
	neighbours = MoveGen(graph, start[0], start[1])
	if len(neighbours) == 0:
		return False, None

	neighbours = sorted(neighbours, key= lambda x: heuristicVal(x, dest, heuristic))
	for neigh in neighbours:
		if neigh not in de:
			return True, neigh

	return True, neighbours[0]


# Exits Game and Prints a Message
def exitGame(message):
	print(message)
	exit()


# Test if the current node is goal node
def GoalTest(curr, node):
	return curr==node


# Generate possible moves
def MoveGen(grid, i, j):
	adjList = list()
	m, n = np.shape(grid)
	if i+1 < m and (grid[i+1, j]==0 or grid[i+1, j] == 2):
		adjList.append([i+1, j])
	if i-1 >= 0 and (grid[i-1, j]==0 or grid[i-1, j] == 2):
		adjList.append([i-1, j])
	if j+1 < n and (grid[i, j+1]==0 or grid[i, j+1] == 2):
		adjList.append([i, j+1])
	if j-1 >= 0 and (grid[i, j-1]==0 or grid[i, j-1] == 2):
		adjList.append([i, j-1])
	return adjList


# Given predecessors of explored nodes, find path from snake head to food 
def FindPath(pred, start, food):
	path = list()
	while True:
		food = pred[food[0]][food[1]]
		path.append(food)
		if food==start:
			break
	return path[::-1]


# Choose an algorithm based on the algorithm choice
def ShortestPath(grid, start, dest, snake, obstacles):
	m, n = np.shape(grid)
	global de
	grid = np.zeros((m, n), dtype=np.int8)
	for x,y in snake:
		grid[x, y] = 1
	for x, y in obstacles:
		grid[x, y] = 3

	visited = np.zeros((m, n), dtype=np.int8)
	
	x = list()
	[x.append([0, 0]) for i in range(n)]
	global beta
	pred = [x.copy() for i in range(m)]
	if algo_choice==0:
		result, pred = BFS(grid, start, dest, pred, visited)
		if result == True:
			path = FindPath(pred, start, dest)
			path.append(dest)
			path = path[1:]
	elif algo_choice==1:
		startTime = time.time()
		result, pred, explored = BestFirstSearch(grid, start, dest, pred, visited, heuristic_choice)
		endTime = time.time()
		if result == True:
			path = FindPath(pred, start, dest)
			path.append(dest)
			path = path[1:]
		print("Path Length: ", len(path))
		print("Explored States: ", explored)
		print("Time Taken: ", endTime-startTime)
		sleep(2)
	elif algo_choice==2:
		result = True
		explored = 0
		while result == True:
			explored += 1
			grid = makeGrid(snake, dest, obstacles)
			result, move = HillClimbingSearch(grid, start, dest, heuristic_choice)
			if move==dest:
				print("Explored Nodes: ", explored)
				sleep(2)
				explored = 0
			if result==False:
				printGrid(grid, snake, obstacles, dest)
				exitGame("Stuck at Local Minima!")
			OneMoveSnake(move, snake, dest, obstacles)
			start = move
	elif algo_choice==3:
		path, result2 = [], True
		while result2 == True:
			grid = makeGrid(snake, dest, obstacles)
			result, move = HillClimbingSearch(grid, start, dest, heuristic_choice)
			if result==False:
				printGrid(grid, snake, obstacles, dest)
				print("Stuck at Local Minima!")
				print("Switching to Denser Function.")
				sleep(1)
				result2, pred, limit, pseudoDest = VariableNeighbourhoodDescent(grid, start, dest, heuristic_choice)
				if result2==False:
					exitGame("Stuck at Global Minima!")
				path = FindPath(pred, start, pseudoDest)
				path = path[1:]
				if result2 == False:
					exitGame("Stuck at Global Minima")
				while len(path) != 0:
					move = path.pop(0)
					OneMoveSnake(move, snake, dest, obstacles)
					start = move
			if result==True:
				OneMoveSnake(move, snake, dest, obstacles)
				start = move

	elif algo_choice==4:
		startTime = time.time()
		result, pred, explored = BeamSearch(grid, start, dest, pred, visited, heuristic_choice, beta)
		endTime = time.time()
		if result == True:
			path = FindPath(pred, start, dest)
			path.append(dest)
			path = path[1:]
		print("Time Taken: ", endTime-startTime)
		try:
			print("Path Length: ", len(path))
		except:
			print("Path Length: 0")
		print("States Explored: ", explored)
		sleep(3)
	elif algo_choice==5:
		result = True
		pathLen = 0
		startTime = time.time()
		while result == True:
			result, move = TabuSearch(grid, start, dest, heuristic_choice)
			pathLen += 1
			if result==False:
				printGrid(grid, snake, obstacles, dest)
				exitGame("Stuck at Local Minima!")
			de.append(move)
			if move==dest:
				endTime = time.time()
				print("Time Taken: ", endTime-startTime)
				print("Path Length: ", pathLen)
				sleep(3)
				pathLen = 0
				startTime = time.time()
			OneMoveSnake(move, snake, dest, obstacles)
			start = move

	if result==False:
		printGrid(grid, snake, obstacles, dest)
		exitGame("GAME OVER!!!\nNo path found!")
	return path


# Move snake given the path to the food
def MoveSnake(path, snake, food, obstacles):
	while len(path) != 0:
		snake = snake[:-1]
		move = path.pop(0)
		snake.insert(0, move)
		grid = makeGrid(snake, food, obstacles)
		printGrid(grid, snake, obstacles, food)
		sleep(delay)
		system('clear') 			
		if snake[0] == food:
			break
		# path = ShortestPath(grid, snake[0], food, snake, obstacles)
	fruitCapture(snake, food, obstacles)
	return


# Move snake given only the next move chosen according to search algorithm
def OneMoveSnake(move, snake, food, obstacles):
	if move in snake and algo_choice!=3:
		exitGame("GAME OVER!!!\nNo path found!")
	if GoalTest(move, food):
		snake.insert(0, move)
		fruitCapture(snake, food, obstacles)
	else:
		del snake[-1]
		snake.insert(0, move)
		grid = makeGrid(snake, food, obstacles)
		printGrid(grid, snake, obstacles, food)
		sleep(delay)
		system('clear') 
	return


# Called after food is captured, new food is generated
def fruitCapture(snake, food, obstacles):
	prev_food = food.copy()
	new_food = randomFoodGeneration(snake, obstacles, prev_food)
	grid = makeGrid(snake, new_food, obstacles)
	while algo_choice==3:
		ShortestPath(grid, snake[0], new_food, snake, obstacles)
	path = ShortestPath(grid, snake[0], new_food, snake, obstacles)
	snake.insert(0, path[0])
	path = path[1:]
	MoveSnake(path, snake, new_food, obstacles)


# Main Function
def main():
	grid = np.zeros((m, n), dtype=np.int8)
	snake = initialSnake(m, n)
	obstacles = plantObstacles(grid, num_obstacles)
	food = randomFoodGeneration(snake, obstacles, [0,0])
	food = [m-1, n-1]
	grid = makeGrid(snake, food, obstacles)
	path = ShortestPath(grid, snake[0], food, snake, obstacles)
	MoveSnake(path, snake, food, obstacles)


main()
