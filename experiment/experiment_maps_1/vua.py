from queue import PriorityQueue
import time
import psutil
import os
import itertools
from scipy.optimize import linear_sum_assignment
import numpy as np

class Direction:
    def __init__(self, vector, char):
        self.vector = vector
        self.char = char

# Define directions with costs (UP, LEFT, DOWN, RIGHT)
U = Direction((0, -1), 'U')
L = Direction((-1, 0), 'L')
D = Direction((0, 1), 'D')
R = Direction((1, 0), 'R')
directions = [U, L, D, R]

INF = int(1e9 + 7)

def read_sokoban_file(filename):
    walls = set()
    goals = set()
    boxes = {}
    player = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        weights = list(map(int, lines[0].strip().split()))
        weight_index = 0
        
        for y, line in enumerate(lines[1:]):
            for x, char in enumerate(line.rstrip()):
                if char == '#':
                    walls.add((x, y))
                elif char == '$':
                    boxes[(x, y)] = weights[weight_index]
                    weight_index += 1
                elif char == '.':
                    goals.add((x, y))
                elif char == '@':
                    player = (x, y)
                elif char == '*':
                    boxes[(x, y)] = weights[weight_index]
                    weight_index += 1
                    goals.add((x, y))
                elif char == '+':
                    player = (x, y)
                    goals.add((x, y))
    
    return walls, goals, boxes, player

def move(player, boxes, cost, direction):
    new_player = (player[0] + direction.vector[0], player[1] + direction.vector[1])
    new_boxes = dict(boxes)
    cost += 1
    is_pushed = False
    if new_player in boxes:
        box_pos = new_player
        new_box_pos = (box_pos[0] + direction.vector[0], box_pos[1] + direction.vector[1])
        box_weight = new_boxes[box_pos]
        del new_boxes[box_pos]
        new_boxes[new_box_pos] = box_weight
        cost += box_weight
        is_pushed = True
    return new_player, frozenset(new_boxes.items()), cost, is_pushed

def get_available_moves(player, boxes, walls):
    available_moves = []
    box_positions = {pos for pos, _ in boxes}
    
    for direction in directions:
        new_pos = (player[0] + direction.vector[0], player[1] + direction.vector[1])
        
        if new_pos not in walls:
            if new_pos in box_positions:
                box_new_pos = (new_pos[0] + direction.vector[0], new_pos[1] + direction.vector[1])
                if box_new_pos not in walls and box_new_pos not in box_positions:
                    available_moves.append(direction)
            else:
                available_moves.append(direction)
    
    return available_moves

def is_win(goals, box_positions):
    return goals.issubset(box_positions)

def calculate_heuristic(player, boxes, goals, available, distance):
    cost_matrix = np.array(
        [[available[i][x][y] * (value + 1) for ((x, y), value) in boxes] for i in range(len(boxes))]
    )

    minValue = cost_matrix[linear_sum_assignment(cost_matrix)].sum()
    if minValue >= INF:
        return INF

    minDistance = INF
    for (x, y), _ in boxes:
        minDistance = min(minDistance, distance[x][y][player[0]][player[1]] - 1)

    return minValue + minDistance

def build(walls, goals, boxes):
    sizex = max(walls, key=lambda x: x[0])[0] + 1
    sizey = max(walls, key=lambda x: x[1])[1] + 1
    k = len(goals)
    available = [[[INF for _ in range(sizey)] for _ in range(sizex)] for _ in range(k)]
    distances = [[[[INF for _ in range(sizey)] for _ in range(sizex)] for _ in range(sizey)] for _ in range(sizex)]

    for fromx in range(sizex):
        for fromy in range(sizey):
            if (fromx, fromy) in walls:
                continue
            queue = [(fromx, fromy)]
            visited = set(queue)
            distances[fromx][fromy][fromx][fromy] = 0
            while queue:
                x, y = queue.pop(0)
                d = distances[fromx][fromy][x][y]
                for direction in directions:
                    new_pos = (x + direction.vector[0], y + direction.vector[1])
                    if new_pos in walls or new_pos in visited or new_pos[0] < 0 or new_pos[0] >= sizex or new_pos[1] < 0 or new_pos[1] >= sizey:
                        continue
                    visited.add(new_pos)
                    queue.append(new_pos)
                    distances[fromx][fromy][new_pos[0]][new_pos[1]] = d + 1

    for k, goal in enumerate(goals):
        queue = [goal]
        visited = set(queue)
        available[k][goal[0]][goal[1]] = 0
        while queue:
            x, y = queue.pop(0)
            d = available[k][x][y]
            for direction in directions:
                new_pos = (x + direction.vector[0], y + direction.vector[1])
                new_player_pos = (new_pos[0] + direction.vector[0], new_pos[1] + direction.vector[1])
                if new_pos in walls or new_pos in visited or new_player_pos in walls:
                    continue
                visited.add(new_pos)
                queue.append(new_pos)
                available[k][new_pos[0]][new_pos[1]] = d + 1

    return distances, available

def A_star(walls, goals, initial_boxes, initial_player):
    initial_memory = get_memory_usage()
    maximum_memory = initial_memory
    distance, available = build(walls, goals, initial_boxes)
    start_time = time.time()
    frontier = PriorityQueue()
    initial_boxes_frozen = frozenset(initial_boxes.items())
    frontier.put((0, initial_player, initial_boxes_frozen, [], []))
    explored = {(initial_player, initial_boxes_frozen): 0}
    heuristic = {(initial_player, initial_boxes_frozen): calculate_heuristic(initial_player, initial_boxes_frozen, goals, available, distance)}
    extended = set()
    while not frontier.empty():
        fake_cost, player, boxes, path, moves = frontier.get()
        boxes_dict = dict(boxes)

        if (player, boxes) in extended:
            continue
        extended.add((player, boxes))
        cost = explored[(player, boxes)]
        
        if is_win(goals, {pos for pos, _ in boxes}):
            end_time = time.time()
            maximum_memory = max(maximum_memory, get_memory_usage())
            return path, moves, end_time - start_time, cost, len(explored), max(0, get_memory_usage() - initial_memory)
            
        for direction in get_available_moves(player, boxes, walls):
            new_player, new_boxes, new_cost, is_push = move(player, boxes_dict, cost, direction)
            state = (new_player, new_boxes)
            move_char = direction.char.upper() if is_push else direction.char.lower()
            
            if state not in explored:
                heuristic[state] = calculate_heuristic(new_player, new_boxes, goals, available, distance)
                explored[state] = new_cost
                if heuristic[state] != INF:
                    frontier.put((new_cost + heuristic[state], new_player, new_boxes, path + [direction.char], moves + [move_char]))
            elif (heuristic[state] != INF) and (new_cost < explored[state]):
                explored[state] = new_cost
                frontier.put((new_cost + heuristic[state], new_player, new_boxes, path + [direction.char], moves + [move_char]))
    
    return None, None, time.time() - start_time, None, len(explored), get_memory_usage() - initial_memory

def get_memory_usage():
    """Get memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    # Read input file
    walls, goals, boxes, player = read_sokoban_file('test_cases/input-10.txt')
    
    # Find solution using UCS
    solution, moves, time_taken, total_cost, nodes_generated, memory_used = A_star(walls, goals, boxes, player)
    
    # Write solution to output file
    with open('A_star.txt', 'w') as f:
        f.write("A*\n")  # Algorithm name
        if solution:
            # Format statistics line
            stats = (
                f"Steps: {len(solution)}, "
                f"Weight: {total_cost}, "
                f"Node: {nodes_generated}, "
                f"Time (ms): {time_taken*1000:.2f}, "
                f"Memory (MB): {memory_used:.2f}"
            )
            f.write(f"{stats}\n")
            # Write move sequence
            f.write(f"{''.join(moves)}\n")
        else:
            stats = (
                f"Steps: -1, "
                f"Weight: -1, "
                f"Node: {nodes_generated}, "
                f"Time (ms): {time_taken*1000:.2f} ms, "
                f"Memory (MB): {memory_used:.2f}"
            )
            f.write(f"{stats}\n")
            f.write("No solution found\n")

if _name_ == '_main_':
    main()
