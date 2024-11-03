import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QComboBox,
    QPushButton,
    QSlider,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QDesktopWidget,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QPainter, QPixmap, QIcon
from math import sqrt
import heapq
import time
import tracemalloc


# region IO Handler
class IOHandler:
    """Input/Output handler class for reading"""

    def __init__(self):
        self.input_file_name = None

    def set_input_file_name(self, input_file_name):
        self.input_file_name = input_file_name

    def parse(self):
        print(f"Reading map from {self.input_file_name}...")
        with open(self.input_file_name, "r") as f:
            rock_weights = [int(weight) for weight in f.readline().strip().split()]
            maze = [[ch for ch in line.replace("\n", "")] for line in f]
            for line in maze:
                print(line)
        return maze, rock_weights

    def write_metrics_result(self, result):
        output_file_name = self.input_file_name.replace("input", "output")
        with open(output_file_name, "a") as f:
            f.write(result)
            print(f"Metrics result is written to {output_file_name}.")


# endregion


# region Problem Formulation
class State:
    """State class for storing the state of the Sokoban puzzle game."""

    def __init__(
        self, maze, rock_weights=None, player_pos=None, rocks_map=None, goals=None
    ):
        self.maze = maze
        self.player_pos = self.find_player() if player_pos is None else player_pos
        self.rocks_map = (
            self.find_rocks(rock_weights) if rocks_map is None else rocks_map
        )
        self.goals = self.find_goals() if goals is None else goals

    def __str__(self):
        return str(self.maze)

    def find_player(self):
        for row_idx, row in enumerate(self.maze):
            for col_idx, _ in enumerate(row):
                if self.maze[row_idx][col_idx] == "@":
                    return row_idx, col_idx

    def find_rocks(self, rock_weights):
        rocks_map = {}
        count = 0
        for row_idx, row in enumerate(self.maze):
            for col_idx, _ in enumerate(row):
                if (
                    self.maze[row_idx][col_idx] == "$"
                    or self.maze[row_idx][col_idx] == "*"
                ):
                    rocks_map[(row_idx, col_idx)] = rock_weights[count]
                    count += 1
        return rocks_map

    def find_goals(self):
        goals = []
        for row_idx, row in enumerate(self.maze):
            for col_idx, _ in enumerate(row):
                if self.maze[row_idx][col_idx] == ".":
                    goals.append((row_idx, col_idx))
        return goals

    def __hash__(self):
        player_pos = self.player_pos
        rocks_map = tuple(sorted(self.rocks_map.items()))
        return hash((player_pos, rocks_map))

    def __eq__(self, other):
        if isinstance(other, State):
            return (
                self.player_pos == other.player_pos
                and self.rocks_map == other.rocks_map
            )
        return False

    def copy(self):
        new_player_pos = (self.player_pos[0], self.player_pos[1])
        new_rocks_map = {k: v for k, v in self.rocks_map.items()}

        return State(self.maze, None, new_player_pos, new_rocks_map, self.goals)


class Node:
    """Node class for search algorithms."""

    def __init__(self, state, parent, action, path_cost, weight_pushed, steps):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.weight_pushed = weight_pushed
        self.steps = steps

    def __hash__(self):
        return hash(self.state)

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state


class Problem:
    """Problem class for Sokoban puzzle game defines the initial state, goal and valid actions."""

    actions = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}

    def __init__(self, initial: State):
        self.initial = initial

    def is_goal(self, state: State):
        for goal in state.goals:
            if goal not in state.rocks_map:
                return False
        return True

    def is_initial(self, state):
        return state.maze == self.initial.maze

    def result(self, state: State, movement: tuple):
        """
        Apply the movement to the given state and return the resulting state.

        Args:
            state (State): The current state of the game, including the maze layout and player position.
            movement (List[int]): A list containing two integers representing the movement in the x and y directions.

        Returns:
            tuple(State, bool, int): A tuple containing the new state object, a boolean (indicating whether the new state is valid), and the cost of moving. If the movement is invalid, returns (None, False).
        """
        # s = time.time()
        new_state = state.copy()
        # e = time.time()
        # print("copy", (e - s) * 1000)
        x, y = new_state.player_pos
        dx, dy = movement
        new_x, new_y = x + dx, y + dy

        if (new_x, new_y) in new_state.rocks_map:
            next_x, next_y = new_x + dx, new_y + dy
            if (next_x, next_y) not in new_state.rocks_map and new_state.maze[next_x][
                next_y
            ] != "#":
                new_state.player_pos = (new_x, new_y)
                new_state.rocks_map[(next_x, next_y)] = new_state.rocks_map[
                    (new_x, new_y)
                ]
                del new_state.rocks_map[(new_x, new_y)]
                return new_state, True, new_state.rocks_map[(next_x, next_y)] + 1
        elif new_state.maze[new_x][new_y] != "#":
            new_state.player_pos = (new_x, new_y)
            return new_state, False, 1

        # Invalid move
        return None, False, 0


# endregion


# region Solver
class Solver:
    def __init__(self, algorithm_name=""):
        self.algorithm_name = algorithm_name

    def change_problem(self, problem: Problem):
        self.problem = problem
        self.steps = 0
        self.total_weight_pushed = 0
        self.total_cost = 0
        self.nodes_generated = 0
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_end = None
        self.result = None

    def start_timer(self):
        self.start_time = time.time()
        tracemalloc.start()
        self.memory_start = tracemalloc.get_traced_memory()[0]

    def stop_timer(self):
        self.end_time = time.time()
        self.memory_end = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

    def solve(self):
        """Override this method in the child class to implement the logic of the algorithm."""
        pass

    def solve_and_measure(self, problem: Problem):
        self.change_problem(problem)
        self.start_timer()
        result = self.solve()
        self.result = "".join(result) if result is not None else None
        self.stop_timer()
        return self.result

    def output_metrics(self):
        if self.result is None:
            return f"{self.algorithm_name}\nNo solution found."

        time_taken = (self.end_time - self.start_time) * 1000
        memory_used = (self.memory_end - self.memory_start) / (1024 * 1024)
        return f"{self.algorithm_name}\nSteps: {self.steps}, Cost: {self.total_cost}, Node: {self.nodes_generated}, Time (ms): {time_taken:.2f}, Memory (MB): {memory_used:.2f}\n{self.result}"

    def trace_path(self, node):
        self.steps = node.steps
        self.total_weight_pushed = node.weight_pushed
        self.total_cost = node.path_cost

        path = []
        while node.parent:
            path.append(node.action)
            node = node.parent

        return path[::-1]


# endregion


# region DFSolver
class DFSolver(Solver):
    def __init__(self):
        super().__init__("DFS")

    def solve(self):
        node = Node(self.problem.initial, None, None, 0, 0, 0)
        if self.problem.is_goal(node.state):
            return self.trace_path(node)

        frontier = [node]
        reached = set()

        reached.add(node.state)
        self.nodes_generated = 1

        while frontier:
            node = frontier.pop()

            for action, movement in self.problem.actions.items():
                child_state, box_moved, moving_cost = self.problem.result(
                    node.state, movement
                )
                if child_state is None:
                    continue

                if child_state not in reached:
                    child_node = Node(
                        child_state,
                        node,
                        action.upper() if box_moved else action,
                        node.path_cost + moving_cost,
                        node.weight_pushed + moving_cost - 1,
                        node.steps + 1,
                    )

                    if self.problem.is_goal(child_state):
                        return self.trace_path(child_node)

                    reached.add(child_state)
                    frontier.append(child_node)
                    self.nodes_generated += 1
        return None


# endregion


# region BFSolver
class BFSolver(Solver):
    def __init__(self):
        super().__init__("BFS")

    def solve(self):
        node = Node(self.problem.initial, None, None, 0, 0, 0)
        if self.problem.is_goal(node.state):
            return self.trace_path(node)

        frontier = []
        reached = set()
        self.nodes_generated = 1

        frontier.append(node)
        reached.add(node.state)

        while frontier:
            node = frontier.pop(0)

            for action, movement in self.problem.actions.items():
                child_state, box_moved, moving_cost = self.problem.result(
                    node.state, movement
                )
                if child_state is None:
                    continue

                child_cost = node.path_cost + moving_cost

                if child_state not in reached:
                    child_node = Node(
                        child_state,
                        node,
                        action.upper() if box_moved else action,
                        child_cost,
                        node.weight_pushed + moving_cost - 1,
                        node.steps + 1,
                    )
                    if self.problem.is_goal(child_state):
                        return self.trace_path(child_node)

                    reached.add(child_state)
                    frontier.append(child_node)
                    self.nodes_generated += 1

        return None


# endregion


# region UCSolver
class UCSolver(Solver):
    def __init__(self):
        super().__init__("UCS")

    def solve(self):
        node = Node(self.problem.initial, None, None, 0, 0, 0)
        frontier = []
        reached = {}  # cost of reaching the node
        self.nodes_generated = 1

        heapq.heappush(frontier, (0, node))

        while frontier:
            _, node = heapq.heappop(frontier)
            state = node.state

            if self.problem.is_goal(state):
                return self.trace_path(node)

            for action, movement in self.problem.actions.items():
                child_state, box_moved, moving_cost = self.problem.result(
                    node.state, movement
                )
                if child_state is None:
                    continue

                child_cost = node.path_cost + moving_cost

                if child_state not in reached or reached[child_state] > child_cost:
                    child_node = Node(
                        child_state,
                        node,
                        action.upper() if box_moved else action,
                        child_cost,
                        node.weight_pushed + moving_cost - 1,
                        node.steps + 1,
                    )
                    reached[child_state] = child_cost
                    heapq.heappush(frontier, (child_cost, child_node))
                    self.nodes_generated += 1

        return None


# endregion


# region AStarSolver
class AStarSolver(Solver):
    def __init__(self):
        super().__init__("A*")
        self.heuristic_measure = 0
        self.heuristic_start = 0
        self.heuristic_calls = 0

    def solve(self):
        node = Node(self.problem.initial, None, None, 0, 0, 0)
        frontier = []
        reached = {}  # combined cost of reaching the node and heuristic cost
        heuristic = {}
        self.nodes_generated = 1

        heapq.heappush(frontier, (0, node))
        heuristic[node.state] = self.heuristic_cost(node.state)
        reached[node.state] = heuristic[node.state]

        while frontier:
            _, node = heapq.heappop(frontier)
            state = node.state

            if self.problem.is_goal(state):
                return self.trace_path(node)

            for action, movement in self.problem.actions.items():
                child_state, box_moved, moving_cost = self.problem.result(
                    node.state, movement
                )
                if child_state is None:
                    continue

                is_child_reached_before = child_state in reached
                if not is_child_reached_before:
                    self.heuristic_start = time.time()
                    heuristic[child_state] = self.heuristic_cost(child_state)
                    self.heuristic_measure += time.time() - self.heuristic_start

                child_combined_cost = (
                    node.path_cost + moving_cost + heuristic[child_state]
                )

                if (
                    not is_child_reached_before
                    or reached[child_state] > child_combined_cost
                ):
                    child_node = Node(
                        child_state,
                        node,
                        action.upper() if box_moved else action,
                        child_combined_cost - heuristic[child_state],
                        node.weight_pushed + moving_cost - 1,
                        node.steps + 1,
                    )
                    reached[child_state] = child_combined_cost
                    heapq.heappush(frontier, (child_combined_cost, child_node))
                    self.nodes_generated += 1

        return None

    def heuristic_cost(self, state):
        """
        Calculate the heuristic cost for the given state.

        Args:
            state (State): The current state of the game, including the maze layout and positions of rocks and goals.

        Returns:
            int: The heuristic cost based on the distance of rocks to their closest goals, weighted by the rock weights.
        """
        heuristic = 0
        goals = state.goals
        rocks = sorted(state.rocks_map.items(), key=lambda x: x[1], reverse=True)
        used_goals = set()

        for rock_pos, rock_weight in rocks:
            min_distance = float("inf")
            closest_goal = None

            for goal in goals:
                if goal not in used_goals:
                    distance = abs(rock_pos[0] - goal[0]) + abs(rock_pos[1] - goal[1])
                    if distance < min_distance:
                        min_distance = distance
                        closest_goal = goal

            if closest_goal:
                used_goals.add(closest_goal)
                heuristic += min_distance * (rock_weight + 1)

            heuristic += abs(state.player_pos[0] - rock_pos[0]) + abs(
                state.player_pos[1] - rock_pos[1]
            )

        self.heuristic_calls += 1
        return heuristic

    def trace_path(self, node):
        self.steps = node.steps
        self.total_weight_pushed = node.weight_pushed
        self.total_cost = node.path_cost

        path = []
        while node.parent:
            path.append(node.action)
            node = node.parent

        print("Heuristic calculation time:", self.heuristic_measure * 1000)
        print("Heuristic calls:", self.heuristic_calls)
        print(
            "Heuristic average time:",
            self.heuristic_measure * 1000 / self.heuristic_calls,
        )

        return path[::-1]


# endregion

MINIMUM_FPS = 1
MAXIMUM_FPS = 100
DEFAULT_FPS = 4


class SokobanVisualizer(QWidget):
    """Visualizer class for Sokoban puzzle game."""

    movements = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}

    # region Visualizer Initialization
    def __init__(self):
        """Initialize the SokobanVisualizer widget."""
        super().__init__()
        self.initialize_variables()
        self.load_images()
        self.prepare_ui()

    def initialize_variables(self):
        """Initialize instance variables with default values."""
        self.tile_size = 40
        self.maze = None
        self.rock_weights = None
        self.rocks_map = {}
        self.moves = []
        self.total_steps = 0
        self.total_cost = 0
        self.speed = 1000 // DEFAULT_FPS
        self.timer = None
        self.player_pos = (0, 0)
        self.player_direction = "d"
        self.move_index = 0
        self.goals = []

    def load_images(self):
        """Load game images from files and store as instance variables."""
        self.wall_image_orig = QPixmap("img/wall.png")
        self.incorrect_box_image_orig = QPixmap("img/incorrect_box.png")
        self.correct_box_image_orig = QPixmap("img/correct_box.png")
        self.goal_image_orig = QPixmap("img/goal.png")
        self.floor_image_orig = QPixmap("img/floor.png")
        self.player_images_orig = {
            "u": QPixmap("img/player_up.png"),
            "d": QPixmap("img/player_down.png"),
            "l": QPixmap("img/player_left.png"),
            "r": QPixmap("img/player_right.png"),
        }
        self.rescale_images()

    def rescale_images(self):
        """Rescale loaded images to match current tile size."""
        self.wall_image = self.wall_image_orig.scaled(self.tile_size, self.tile_size)
        self.incorrect_box_image = self.incorrect_box_image_orig.scaled(
            self.tile_size, self.tile_size
        )
        self.correct_box_image = self.correct_box_image_orig.scaled(
            self.tile_size, self.tile_size
        )
        self.goal_image = self.goal_image_orig
        self.floor_image = self.floor_image_orig.scaled(self.tile_size, self.tile_size)
        self.player_images = {
            direction: image.scaledToHeight(self.tile_size)
            for direction, image in self.player_images_orig.items()
        }

    # endregion

    # region Visualizer Map Management
    def change_map(self, maze, rock_weights):
        """Update the current map with new maze and rock weights."""
        self.maze = maze
        self.rock_weights = rock_weights
        self.reset_map()
        self.prepare_ui()
        self.update()

    def reset_map(self):
        """Reset the map to its initial state."""
        self.load_maze_data()
        self.move_index = 0
        self.total_cost = 0
        self.total_steps = 0
        self.update()

    def load_maze_data(self):
        """Load maze data including rocks and player positions."""
        self.rocks_map = {}
        count = 0
        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                if cell == "$":
                    self.rocks_map[(i, j)] = self.rock_weights[count]
                    count += 1
                elif cell == "@":
                    self.player_pos = (i, j)
        self.goals = [
            (i, j)
            for i, row in enumerate(self.maze)
            for j, cell in enumerate(row)
            if cell == "."
        ]

    # endregion

    # region Visualizer UI Setup and Drawing
    def prepare_ui(self):
        """Set up the UI dimensions and scaling."""
        if not self.maze:
            return

        self.reset_map()
        self.width = max(len(row) for row in self.maze) * self.tile_size
        self.height = len(self.maze) * self.tile_size
        self.setMinimumSize(self.width, self.height)
        self.update_player_direction("d")

        tile_size = int(sqrt(800 * 600 / (self.width * self.height)))
        self.tile_size = 40 if tile_size > 40 or tile_size < 30 else tile_size

        if self.tile_size != 40:
            self.rescale_images()

    def paintEvent(self, event):
        """Handle paint events for the widget."""
        painter = QPainter(self)
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] == "#":
                    self.paint_cell(j, i, "#", painter)
                elif self.maze[i][j] == "." or self.maze[i][j] == "*":
                    self.paint_cell(j, i, ".", painter)

        self.paint_cell(self.player_pos[1], self.player_pos[0], "@", painter)

        for rock_pos, _ in self.rocks_map.items():
            cell = self.maze[rock_pos[0]][rock_pos[1]]
            self.paint_cell(
                rock_pos[1],
                rock_pos[0],
                "*" if cell == "." or cell == "*" else "$",
                painter,
            )
        painter.end()

    def paint_cell(self, x, y, cell, painter):
        """Paint a single cell in the maze."""
        tile_x = x * self.tile_size
        tile_y = y * self.tile_size
        opacity = 1.0

        if cell == "#":
            image = self.wall_image
        elif cell == ".":
            image = self.goal_image
        elif cell == "$":
            image = self.incorrect_box_image
        elif cell == "*":
            image = self.correct_box_image
        elif cell == "@":
            image = self.player_images[self.player_direction]
        elif cell == "+":
            image = self.player_images[self.player_direction]
            opacity = 0.5
        else:
            return

        iw = image.width()
        ih = image.height()
        offset_x = (self.tile_size - iw) // 2
        offset_y = (self.tile_size - ih) // 2
        pos = QPoint(tile_x + offset_x, tile_y + offset_y)

        painter.setOpacity(opacity)
        painter.drawPixmap(pos, image)

        if cell == "$" or cell == "*":
            self.paint_rock_text(x, y, str(self.rocks_map[(y, x)]), painter)

    def paint_rock_text(self, x, y, text, painter):
        """Paint the weight text on rocks."""
        painter.setPen(Qt.white)
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)

        text_rect = painter.fontMetrics().boundingRect(text)
        text_x = x * self.tile_size + (self.tile_size - text_rect.width()) // 2
        text_y = y * self.tile_size + (self.tile_size - text_rect.height()) // 2 - 5

        painter.setPen(Qt.black)
        painter.drawText(text_x + 1, text_y + text_rect.height(), text)
        painter.setPen(Qt.white)
        painter.drawText(text_x, text_y + text_rect.height(), text)

    # endregion

    # region Visualizer Game Logic
    def move_player(self, action):
        """Handle player movement and rock pushing."""
        dx, dy = self.movements[action]
        new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy

        if (new_x, new_y) in self.rocks_map:
            box_x, box_y = new_x + dx, new_y + dy
            if (box_x, box_y) not in self.rocks_map and self.maze[box_x][box_y] != "#":
                self.player_pos = (new_x, new_y)
                self.rocks_map[(box_x, box_y)] = self.rocks_map.pop((new_x, new_y))
                self.total_cost += self.rocks_map[(box_x, box_y)] + 1
        elif self.maze[new_x][new_y] != "#":
            self.player_pos = (new_x, new_y)
            self.total_cost += 1

        self.total_steps += 1

    def update_player_direction(self, direction):
        """Update the player's facing direction."""
        self.player_direction = direction

    def update_game(self):
        """Update game state for each animation frame."""
        if self.move_index < len(self.moves):
            action = self.moves[self.move_index].lower()
            self.update_player_direction(action)
            self.move_player(action)
            self.move_index += 1

            if hasattr(self.parent(), "update_status_labels"):
                self.parent().update_status_labels(self.total_steps, self.total_cost)

            self.update()
        else:
            self.timer.stop()
            if hasattr(self.parent(), "visualization_complete"):
                self.parent().visualization_complete()

    # endregion

    # region Visualizer Control
    def set_moves(self, moves):
        """Set the sequence of moves to visualize."""
        self.moves = moves
        self.move_index = 0

    def start_visualization(self):
        """Start the visualization animation."""
        if not self.moves:
            return
        if self.timer is None:
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_game)
        self.timer.start(self.speed)

    def pause_visualization(self):
        """Pause the visualization animation."""
        if self.timer and self.timer.isActive():
            self.timer.stop()

    def resume_visualization(self):
        """Resume the paused visualization."""
        if self.timer and not self.timer.isActive():
            self.timer.start(self.speed)

    def reset_visualization(self):
        """Reset visualization to initial state."""
        self.reset_map()
        self.update_player_direction("d")
        self.update()
        if self.timer:
            self.timer.stop()

    def change_speed(self, fps):
        """Change visualization animation speed."""
        self.speed = 1000 // fps
        if self.timer and self.timer.isActive():
            self.timer.start(self.speed)

    # endregion


class App(QWidget):
    """Main application window for Sokoban puzzle solver and visualizer"""

    # region App Initialization
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ares's Adventure")
        self.setWindowIcon(QIcon("img/icon.png"))
        self.initialize_components()
        self.init_ui()
        self.center()

    def initialize_components(self):
        """Initialize core application components"""
        self.io_handler = IOHandler()
        self.maze = []
        self.rock_weights = []
        self.solvers = {
            "DFS": DFSolver(),
            "BFS": BFSolver(),
            "UCS": UCSolver(),
            "A*": AStarSolver(),
        }
        self.results = {}
        self.visualizer = None

    # endregion

    # region App UI Setup
    def init_ui(self):
        """Initialize user interface components"""
        self.create_controls()
        self.create_layouts()
        self.setup_initial_state()

    def create_controls(self):
        """Create UI control elements"""
        self.create_dropdowns()
        self.create_buttons()
        self.create_status_labels()
        self.create_speed_slider()
        self.create_speed_slider_label()
        self.visualizer = SokobanVisualizer()

    def create_dropdowns(self):
        """Create and configure dropdown menus"""
        self.map_dropdown = QComboBox(self)
        input_files = [
            f for f in os.listdir(".") if f.endswith(".txt") and "input" in f
        ]
        self.map_dropdown.addItems(input_files)
        self.map_dropdown.currentIndexChanged.connect(self.update_map)

        self.algorithm_dropdown = QComboBox(self)
        self.algorithm_dropdown.addItems(self.solvers.keys())

    def create_buttons(self):
        """Create control buttons"""
        self.solve_button = QPushButton("Solve Map!", self)
        self.solve_button.clicked.connect(self.run_solver)

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_visualization)

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_visualization)

        self.reset_button = QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.reset_visualization)

    def create_status_labels(self):
        """Create status display labels"""
        self.steps_label = QLabel("Steps: 0")
        self.cost_label = QLabel("Cost: 0")

    def create_speed_slider(self):
        """Create speed control slider"""
        self.speed_slider = QSlider(Qt.Horizontal, self)
        self.speed_slider.setMinimum(MINIMUM_FPS)
        self.speed_slider.setMaximum(MAXIMUM_FPS)
        self.speed_slider.setValue(DEFAULT_FPS)
        self.speed_slider.valueChanged.connect(self.change_speed)

    def create_speed_slider_label(self):
        """Create speed control slider text"""
        self.speed_slider_text = QLabel(f"{DEFAULT_FPS} FPS")

    def create_layouts(self):
        """Create and configure layouts"""
        main_layout = QVBoxLayout()
        main_layout.addLayout(self.create_top_layout())
        main_layout.addWidget(self.visualizer)
        main_layout.addLayout(self.create_status_layout())
        main_layout.addLayout(self.create_bottom_layout())
        self.setLayout(main_layout)

    def create_top_layout(self):
        """Create top control layout"""
        layout = QHBoxLayout()
        layout.addWidget(self.map_dropdown)
        layout.addWidget(self.algorithm_dropdown)
        layout.addWidget(self.solve_button)
        return layout

    def create_status_layout(self):
        """Create status display layout"""
        layout = QHBoxLayout()
        layout.addWidget(self.steps_label)
        layout.addWidget(self.cost_label)
        layout.addStretch()
        return layout

    def create_bottom_layout(self):
        """Create bottom control layout"""
        layout = QHBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.reset_button)
        layout.addStretch()
        layout.addWidget(QLabel("Speed:"))
        layout.addWidget(self.speed_slider)
        layout.addWidget(self.speed_slider_text)
        return layout

    def setup_initial_state(self):
        """Setup initial application state"""
        self.enable_visualization(False)
        self.update_map()
        self.show()

    # endregion

    # region App Event Handlers
    def update_map(self):
        """Update current map and reset visualization"""
        map_file = self.map_dropdown.currentText()
        if not map_file:
            return

        self.io_handler.set_input_file_name(map_file)
        self.maze, self.rock_weights = self.io_handler.parse()
        self.visualizer.change_map(self.maze, self.rock_weights)
        self.reset_ui_state()
        self.adjust_window()

    def run_solver(self):
        """Execute selected solver algorithm"""
        self.update_map()
        algorithm = self.algorithm_dropdown.currentText()

        try:
            problem = Problem(State(self.maze, self.rock_weights))
            solver = self.solvers[algorithm]
            result = solver.solve_and_measure(problem)

            # Store and update visualizer with results
            self.results[algorithm] = result
            if result:
                self.visualizer.set_moves(result)
                self.process_solver_results(solver)
                self.enable_visualization(True)
            else:
                QMessageBox.information(self, "Information", "No solution found.")
                self.enable_visualization(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Solver failed: {str(e)}")

    # endregion

    # region App Visualization Control
    def start_visualization(self):
        """Start or resume visualization"""
        algorithm = self.algorithm_dropdown.currentText()
        if not self.validate_visualization(algorithm):
            return

        self.visualizer.start_visualization()
        self.update_control_states(False)

    def pause_visualization(self):
        """Pause ongoing visualization"""
        self.visualizer.pause_visualization()
        self.update_control_states(True)
        self.start_button.setText("Resume")

    def reset_visualization(self):
        """Reset visualization to initial state"""
        self.visualizer.reset_visualization()
        self.update_control_states(True)
        self.start_button.setText("Start")
        self.update_status_labels(0, 0)

    # endregion

    # region App UI Helpers
    def update_status_labels(self, steps, cost):
        """Update status display values"""
        self.steps_label.setText(f"Steps: {steps}")
        self.cost_label.setText(f"Cost: {cost}")

    def center(self):
        """Center window on screen"""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def adjust_window(self):
        """Adjust and center window"""
        self.adjustSize()
        self.center()

    def reset_ui_state(self):
        """Reset UI to initial state"""
        self.enable_visualization(False)
        self.update_status_labels(0, 0)

    def enable_visualization(self, enable):
        """Update visualization control states"""
        self.start_button.setEnabled(enable)
        self.pause_button.setEnabled(False)
        self.reset_button.setEnabled(enable)
        self.start_button.setText("Start")

    def update_control_states(self, enable_start):
        """Update control states based on visualization state"""
        self.start_button.setEnabled(enable_start)
        self.pause_button.setEnabled(not enable_start)
        self.solve_button.setEnabled(enable_start)
        self.map_dropdown.setEnabled(enable_start)
        self.algorithm_dropdown.setEnabled(enable_start)

    def change_speed(self, fps):
        """Update visualization speed"""
        self.visualizer.change_speed(fps)
        self.speed_slider_text.setText(f"{fps} FPS")

    def visualization_complete(self):
        """Handle visualization completion"""
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(False)

    # endregion

    # region App Helpers
    def validate_visualization(self, algorithm):
        """Validate visualization can start"""
        if algorithm not in self.results:
            QMessageBox.information(self, "Information", "Please run the solver first.")
            return False
        if self.results[algorithm] is None:
            QMessageBox.information(
                self, "Information", f"No solution found for {algorithm}"
            )
            return False
        return True

    def process_solver_results(self, solver):
        """Process and display solver results"""
        output_result = solver.output_metrics() + "\n"
        print(output_result)
        self.io_handler.write_metrics_result(output_result)
        QMessageBox.information(self, "Information", "Map solved successfully.")

    # endregion


def main():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
