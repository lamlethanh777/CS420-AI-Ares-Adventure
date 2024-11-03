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
from PyQt5.QtGui import (
    QPainter,
    QColor,
    QBrush,
    QPixmap,
    QLinearGradient,
    QPainterPath,
    QPen,
    QFont,
)
from math import sqrt
import heapq
import time
import tracemalloc


class IOHandler:
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
        with open(output_file_name, "w") as f:
            f.write(result)
            print(f"Metrics result is written to {output_file_name}.")


class State:
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
    movement = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}
    actions = ["u", "d", "l", "r"]

    def __init__(self, initial: State):
        self.initial = initial

    def is_goal(self, state: State):
        for goal in state.goals:
            if goal not in state.rocks_map:
                return False
        return True

    def is_initial(self, state):
        return state.maze == self.initial.maze

    def result(self, state: State, action: str):
        """
        Determines the resulting state and action after applying a given action to the current state.

        Args:
            state (object): The current state of the environment.
            action (str): The action to be applied.

        Returns:
            tuple(State, str, int): A tuple containing the new state, the action taken, and the cost of moving. If the action results in moving a box, the action is returned in uppercase. If the action cannot be applied, returns (None, None).
        """
        return self.apply(state, self.actions[action])

    def apply(self, state: State, movement: tuple):
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

    def solve(self, problem: Problem):
        """Override this method in the child class to implement the logic of the algorithm."""
        pass

    def solve_and_measure(self, problem: Problem):
        self.start_timer()
        result = self.solve(problem)
        self.result = "".join(result) if result is not None else None
        self.stop_timer()
        return self.result

    def output_metrics(self):
        if self.result is None:
            return f"{self.algorithm_name}\nNo solution found."

        time_taken = (self.end_time - self.start_time) * 1000
        memory_used = (self.memory_end - self.memory_start) / (1024 * 1024)
        return f"{self.algorithm_name}\nSteps: {self.steps}, Cost: {self.total_cost}, Node: {self.nodes_generated}, Time (ms): {time_taken:.2f}, Memory (MB): {memory_used:.2f}\n{self.result}"


class AStarSolver(Solver):
    def __init__(self):
        super().__init__("A*")
        self.heuristic_measure = 0
        self.heuristic_start = 0
        self.heuristic_calls = 0

    def solve(self, problem: Problem):
        super().solve(problem)

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

            for action in self.problem.actions:
                child_state, box_moved, moving_cost = self.problem.apply(
                    node.state, problem.movement[action]
                )
                if child_state is None:
                    continue

                is_child_reached_before = child_state in reached
                if not is_child_reached_before:
                    # self.heuristic_start = time.time()
                    heuristic[child_state] = self.heuristic_cost(child_state)
                    # self.heuristic_measure += time.time() - self.heuristic_start

                child_combined_cost = (
                    node.path_cost + moving_cost + heuristic[child_state]
                )

                if (
                    not is_child_reached_before
                    or reached[child_state] > child_combined_cost
                ):
                    reached[child_state] = child_combined_cost

                    child_node = Node(
                        child_state,
                        node,
                        action.upper() if box_moved else action,
                        child_combined_cost - heuristic[child_state],
                        node.weight_pushed + moving_cost - 1,
                        node.steps + 1,
                    )
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


class DFSolver(Solver):
    def __init__(self):
        super().__init__("DFS")

    # TODO: Logic of DFS algorithm is implemented here
    def solve(self, problem: Problem):
        super().solve(problem)
        return None


class BFSolver(Solver):
    def __init__(self):
        super().__init__("BFS")

    # TODO: Logic of BFS algorithm is implemented here
    # Hint: You should use the heapq to implement the frontier
    def solve(self, problem: Problem):
        super().solve(problem)
        return None


class UCSolver(Solver):
    def __init__(self):
        super().__init__("UCS")

    # TODO: Logic of UCS algorithm is implemented here
    def solve(self, problem: Problem):
        super().solve(problem)
        return None


MINIMUM_FPS = 1
MAXIMUM_FPS = 100
DEFAULT_FPS = 10


class SokobanVisualizer(QWidget):
    movements = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}

    def __init__(self):
        super().__init__()

        # Initialize variables
        self.tile_size = 40
        self.maze = None
        self.rock_weights = None
        self.rocks_map = None
        self.update_cells_list = []
        self.moves = []
        self.total_steps = 0
        self.total_cost = 0

        self.speed = 1000 // DEFAULT_FPS
        self.timer = None

        # Initialize UI elements
        self.load_images()
        self.prepare_ui()

    def change_map(self, maze, rock_weights):
        self.change_map_data(maze, rock_weights)
        self.prepare_ui()
        self.update()

    def load_images(self):
        # Load and scale images
        self.wall_image = QPixmap("img/wall.png").scaled(self.tile_size, self.tile_size)
        self.incorrect_box_image = QPixmap("img/incorrect_box.png").scaled(
            self.tile_size, self.tile_size
        )
        self.correct_box_image = QPixmap("img/correct_box.png").scaled(
            self.tile_size, self.tile_size
        )
        self.goal_image = QPixmap("img/goal.png")
        self.floor_image = QPixmap("img/floor.png").scaled(
            self.tile_size, self.tile_size
        )
        # Load player images for each direction
        self.player_images = {
            "u": QPixmap("img/player_up.png").scaledToHeight(self.tile_size),
            "d": QPixmap("img/player_down.png").scaledToHeight(self.tile_size),
            "l": QPixmap("img/player_left.png").scaledToHeight(self.tile_size),
            "r": QPixmap("img/player_right.png").scaledToHeight(self.tile_size),
        }

    def change_map_data(self, maze, rock_weights):
        self.maze = maze
        self.rock_weights = rock_weights
        self.rocks_map = {}
        count = 0
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if maze[i][j] == "$" or maze[i][j] == "*":
                    self.rocks_map[(i, j)] = rock_weights[count]
                    count += 1

        self.player_pos = next(
            (i, j)
            for i in range(len(maze))
            for j in range(len(maze[i]))
            if maze[i][j] == "@"
        )
        self.goals = [
            (i, j)
            for i in range(len(maze))
            for j in range(len(maze[i]))
            if maze[i][j] == "."
        ]
        self.move_index = 0
        self.total_cost = 0
        self.total_steps = 0
        self.update()

    def rescale_images(self):
        self.wall_image = self.wall_image.scaled(self.tile_size, self.tile_size)
        self.incorrect_box_image = self.incorrect_box_image.scaled(
            self.tile_size, self.tile_size
        )
        self.correct_box_image = self.correct_box_image.scaled(
            self.tile_size, self.tile_size
        )
        self.goal_image = self.goal_image.scaled(self.tile_size, self.tile_size)
        self.floor_image = self.floor_image.scaled(self.tile_size, self.tile_size)
        self.player_images = {
            "u": self.player_images["u"].scaledToHeight(self.tile_size),
            "d": self.player_images["d"].scaledToHeight(self.tile_size),
            "l": self.player_images["l"].scaledToHeight(self.tile_size),
            "r": self.player_images["r"].scaledToHeight(self.tile_size),
        }

    def reset_map(self):
        self.change_map_data(self.maze, self.rock_weights)

    def prepare_ui(self):
        if not self.maze:
            return

        self.reset_map()

        self.width = max(len(row) for row in self.maze) * self.tile_size
        self.height = len(self.maze) * self.tile_size

        self.setMinimumSize(self.width, self.height + 40)
        self.update_player_direction("d")

        tile_size = int(sqrt(800 * 600 / (self.width * self.height)))
        self.tile_size = 40 if tile_size > 40 or tile_size < 30 else tile_size

        if self.tile_size != 40:
            self.rescale_images()

    def update_player_direction(self, direction):
        self.player_direction = direction

    def paint_rock_text(self, x, y, text, painter):
        # Set text properties
        painter.setPen(Qt.white)
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)

        # Calculate text rectangle in the middle of the box
        text_rect = painter.fontMetrics().boundingRect(text)
        text_x = x * self.tile_size + (self.tile_size - text_rect.width()) // 2
        text_y = y * self.tile_size + (self.tile_size - text_rect.height()) // 2 - 5

        # Draw text with black background for better visibility
        painter.setPen(Qt.black)
        painter.drawText(text_x + 1, text_y + text_rect.height(), text)
        painter.setPen(Qt.white)
        painter.drawText(text_x, text_y + text_rect.height(), text)

    def paint_cell(self, x, y, cell, painter):
        tile_x = x * self.tile_size
        tile_y = y * self.tile_size
        
        opacity = 1.0

        # Determine which image to draw
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

        # Calculate the position to center the image
        iw = image.width()
        ih = image.height()
        offset_x = (self.tile_size - iw) // 2
        offset_y = (self.tile_size - ih) // 2
        pos = QPoint(tile_x + offset_x, tile_y + offset_y)

        painter.setOpacity(opacity)
        painter.drawPixmap(pos, image)

        if cell == "$" or cell == "*":
            self.paint_rock_text(x, y, str(self.rocks_map[(y, x)]), painter)

    def paint_cost(self, painter):
        # Status bar dimensions
        bar_height = 30
        bar_y = self.height - bar_height
        bar_width = self.width
        button_height = 24
        button_padding = 15
        button_spacing = 20
        corner_radius = 4

        # Draw main status bar background
        painter.fillRect(0, bar_y, bar_width, bar_height, QColor(50, 50, 50, 200))

        # Configure text style
        font = QFont("Segoe UI", 10)
        font.setBold(True)
        painter.setFont(font)

        # Button style parameters
        button_y = bar_y + (bar_height - button_height) // 2

        # Draw Steps button
        steps_text = f"Steps: {self.total_steps}"
        text_width = painter.fontMetrics().width(steps_text)
        button_width = text_width + 2 * button_padding

        # Left button (Steps)
        left_x = 10
        gradient = QLinearGradient(left_x, button_y, left_x, button_y + button_height)
        gradient.setColorAt(0, QColor(240, 240, 240))
        gradient.setColorAt(1, QColor(224, 224, 224))

        path = QPainterPath()
        path.addRoundedRect(
            left_x, button_y, button_width, button_height, corner_radius, corner_radius
        )
        painter.fillPath(path, gradient)

        # Draw button border
        painter.setPen(QPen(QColor(204, 204, 204)))
        painter.drawRoundedRect(
            left_x, button_y, button_width, button_height, corner_radius, corner_radius
        )

        # Draw Steps text
        painter.setPen(Qt.black)
        text_y = (
            button_y
            + (
                button_height
                + painter.fontMetrics().ascent()
                - painter.fontMetrics().descent()
            )
            // 2
        )
        painter.drawText(left_x + button_padding, text_y, steps_text)

        # Right button (Cost)
        cost_text = f"Cost: {self.total_cost}"
        text_width = painter.fontMetrics().width(cost_text)
        button_width = text_width + 2 * button_padding
        right_x = left_x + button_width + button_spacing

        gradient = QLinearGradient(right_x, button_y, right_x, button_y + button_height)
        gradient.setColorAt(0, QColor(240, 240, 240))
        gradient.setColorAt(1, QColor(224, 224, 224))

        path = QPainterPath()
        path.addRoundedRect(
            right_x, button_y, button_width, button_height, corner_radius, corner_radius
        )
        painter.fillPath(path, gradient)

        # Draw button border
        painter.setPen(QPen(QColor(204, 204, 204)))
        painter.drawRoundedRect(
            right_x, button_y, button_width, button_height, corner_radius, corner_radius
        )

        # Draw Cost text
        painter.setPen(Qt.black)
        painter.drawText(right_x + button_padding, text_y, cost_text)

    def paintEvent(self, event):
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

        self.paint_cost(painter)

        painter.end()

    def move_player(self, action):
        # Implement player movement logic
        dx, dy = self.movements[action]
        new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy

        if (new_x, new_y) in self.rocks_map:
            box_x, box_y = new_x + dx, new_y + dy
            if (box_x, box_y) not in self.rocks_map and self.maze[box_x][box_y] != "#":
                self.player_pos = (new_x, new_y)
                self.rocks_map[(box_x, box_y)] = self.rocks_map.pop((new_x, new_y))
                self.total_steps += 1
                self.total_cost += self.rocks_map[(box_x, box_y)] + 1
        elif self.maze[new_x][new_y] != "#":
            self.player_pos = (new_x, new_y)
            self.total_steps += 1
            self.total_cost += 1

    def set_moves(self, moves):
        self.moves = moves
        self.move_index = 0

    def update_game(self):
        if self.move_index < len(self.moves):
            action = self.moves[self.move_index].lower()

            # Update the player direction based on action
            self.update_player_direction(action)

            # Implement movement logic
            self.move_player(action)

            self.move_index += 1
            self.update()
        else:
            self.timer.stop()

    def start_visualization(self):
        """Start the game loop to visualize moves."""
        if not self.moves:
            return
        if self.timer is None:
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_game)
        self.timer.start(self.speed)

    def pause_visualization(self):
        """Pause the visualization."""
        if self.timer and self.timer.isActive():
            self.timer.stop()

    def reset_visualization(self):
        """Reset the visualization to the initial state."""
        self.reset_map()
        self.update_player_direction("d")
        self.update()
        if self.timer:
            self.timer.stop()

    def change_speed(self, fps):
        """Change the speed of the visualization."""
        self.speed = 1000 // fps
        if self.timer and self.timer.isActive():
            self.timer.start(self.speed)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ares's Adventure")

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

        self.is_visualization_enabled = False
        self.is_started = False

        self.visualizer = None
        self.init_ui()

    def init_ui(self):
        # Dropdown for maps
        self.map_dropdown = QComboBox(self)
        input_files = [
            f for f in os.listdir(".") if f.endswith(".txt") and "input" in f
        ]
        self.map_dropdown.addItems(input_files)
        self.map_dropdown.currentIndexChanged.connect(self.change_map)

        # Dropdown for algorithms
        self.algorithm_dropdown = QComboBox(self)
        self.algorithm_dropdown.addItems(self.solvers.keys())

        # Run button
        self.run_button = QPushButton("Run", self)
        self.run_button.clicked.connect(self.run_solvers)

        # Start, Pause, Reset buttons
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_visualization)
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_visualization)
        self.reset_button = QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.reset_visualization)

        # Slider for speed
        self.speed_slider = QSlider(Qt.Horizontal, self)
        self.speed_slider.setMinimum(MINIMUM_FPS)
        self.speed_slider.setMaximum(MAXIMUM_FPS)
        self.speed_slider.setValue(DEFAULT_FPS)
        self.speed_slider.valueChanged.connect(self.change_speed)

        # Layouts
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.map_dropdown)
        top_layout.addWidget(self.run_button)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.algorithm_dropdown)
        bottom_layout.addWidget(self.start_button)
        bottom_layout.addWidget(self.pause_button)
        bottom_layout.addWidget(self.reset_button)
        bottom_layout.addStretch()
        bottom_layout.addWidget(QLabel("Speed:"))
        bottom_layout.addWidget(self.speed_slider)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)

        # Visualization area
        self.visualizer = SokobanVisualizer()
        main_layout.addWidget(self.visualizer)

        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

        # Load the initial map
        self.change_map()

        self.show()
        self.center()  # Center the app window on the screen

    def enable_visualization(self, enable):
        self.is_visualization_enabled = enable
        self.algorithm_dropdown.setEnabled(enable)
        self.start_button.setEnabled(enable)
        self.pause_button.setEnabled(enable)
        self.reset_button.setEnabled(enable)

    def center(self):
        """Centers the window on the screen."""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def change_map(self):
        map_file = self.map_dropdown.currentText()
        print(map_file)
        if not map_file:
            return

        # Read the map file
        self.io_handler.set_input_file_name(map_file)
        self.maze, self.rock_weights = self.io_handler.parse()

        # Update the visualizer
        self.visualizer.change_map(self.maze, self.rock_weights)
        self.enable_visualization(False)

        # Adjust the size of the main window to fit the new content
        self.adjustSize()
        # Center the window after resizing
        self.center()

    def run_solvers(self):
        if not self.maze:
            QMessageBox.information(self, "Information", "Please select the map first!")
        # algorithm = self.algorithm_dropdown.currentText()
        print("Running solvers...")
        problem = Problem(State(self.maze, self.rock_weights))

        # Run the solver and get moves
        output_result = ""
        for name, solver in self.solvers.items():
            solver.change_problem(problem)
            self.results[name] = solver.solve_and_measure(problem)
            output_result += solver.output_metrics() + "\n"
        print(output_result)
        self.io_handler.write_metrics_result(output_result)

        QMessageBox.information(self, "Information", "Solver completed successfully.")

        # Enable visualization
        self.enable_visualization(True)

    def start_visualization(self):
        if self.is_started:
            return

        self.is_started = True
        algorithm = self.algorithm_dropdown.currentText()
        if algorithm in self.results:
            moves = self.results[algorithm]
            self.visualizer.set_moves(moves)
            self.visualizer.start_visualization()
        else:
            QMessageBox.information(
                self, "Information", "Please run the solvers first."
            )

    def pause_visualization(self):
        """Pause the ongoing visualization."""
        if self.is_started:
            self.is_started = False
            self.visualizer.pause_visualization()

    def reset_visualization(self):
        """Reset the visualization to the initial state."""
        self.is_started = False
        self.visualizer.reset_visualization()

    def change_speed(self, fps):
        """Change the speed of the visualization."""
        self.visualizer.change_speed(fps)


def main():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
