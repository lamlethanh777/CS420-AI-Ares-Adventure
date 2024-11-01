import pygame
from queue import PriorityQueue
from queue import Queue
import copy
import time
import tracemalloc
import pygame_widgets
from pygame_widgets.button import Button
from pygame_widgets.dropdown import Dropdown
from pygame_widgets.slider import Slider
import os


class IOHandler:
    def __init__(self):
        self.input_file_name = None

    def set_input_file_name(self, input_file_name):
        self.input_file_name = input_file_name

    def parse(self):
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


class Visualizer:
    FPS = 30

    # Colors
    COLOR_BG = (150, 150, 150)  # Gray background
    COLOR_WALL = (139, 69, 19)  # Brown for walls
    COLOR_GOAL = (255, 105, 180)  # Hot pink for goals
    COLOR_ARES = (0, 255, 0)  # Green for Ares
    COLOR_BOX = (105, 105, 105)  # Gray for boxes
    COLOR_BOX_WEIGHT = (0, 0, 0)  # Black for box weights
    COLOR_TEXT = (255, 255, 255)  # White for text
    COLOR_OCCUPIED = (183, 224, 255)  # occupied goals
    COLOR_BUTTON = (0, 0, 255)  # Blue for buttons

    def __init__(self):
        # self.change_map(maze, rock_weights, moves)

        pygame.init()
        pygame.display.set_caption("Ares's Adventure")

        # Get full screen size and set to half
        info = pygame.display.Info()
        self.WIDTH = info.current_w // 4
        self.HEIGHT = info.current_h // 4

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.Font("freesansbold.ttf", 18)
        self.clock = pygame.time.Clock()

    def change_map(self, maze, rock_weights, moves):
        self.original_maze = maze
        self.rock_weights = rock_weights
        self.moves = moves
        self.reset_game()

        # Calculate new tile size
        self.TILE_SIZE = min(self.WIDTH // len(maze[0]), self.HEIGHT // len(maze))

    def reset_game(self):
        self.maze = [list(row) for row in self.original_maze]
        self.moves = self.moves
        self.goal_positions = [
            (x, y)
            for x, row in enumerate(self.maze)
            for y, tile in enumerate(row)
            if tile == "."
        ]
        self.player_pos = self.find_player()
        self.running = False
        self.paused = False
        self.move_index = 0
        self.speed = 0.25  # Set initial speed to be slow
        self.cost = 0
        self.steps = 0
        self.rocks_map = {}
        count = 0
        for x, row in enumerate(self.maze):
            for y, tile in enumerate(row):
                if tile == "$":
                    self.rocks_map[(x, y)] = self.rock_weights[count]
                    count += 1
        print(self.rocks_map)

    def find_player(self):
        for row_idx, row in enumerate(self.maze):
            if "@" in row:
                return row_idx, row.index("@")
        return None

    def draw_maze(self):
        for row_idx, row in enumerate(self.maze):
            for col_idx, tile in enumerate(row):
                x = col_idx * self.TILE_SIZE
                y = row_idx * self.TILE_SIZE

                if tile == "#":
                    pygame.draw.rect(
                        self.screen,
                        self.COLOR_WALL,
                        (x, y, self.TILE_SIZE, self.TILE_SIZE),
                    )
                elif tile == "$":
                    text_rect = pygame.draw.rect(
                        self.screen,
                        self.COLOR_BOX,
                        (x + 10, y + 10, self.TILE_SIZE - 20, self.TILE_SIZE - 20),
                    )
                    self.draw_text(
                        str(self.get_rock_weight(row_idx, col_idx)),
                        text_rect,
                        self.COLOR_BOX_WEIGHT,
                        self.COLOR_BOX,
                    )
                elif tile == "@" or tile == "+":
                    pygame.draw.circle(
                        self.screen,
                        self.COLOR_ARES,
                        (x + self.TILE_SIZE // 2, y + self.TILE_SIZE // 2),
                        self.TILE_SIZE // 3,
                    )
                elif tile == "*":
                    text_rect = pygame.draw.circle(
                        self.screen,
                        self.COLOR_OCCUPIED,
                        (x + self.TILE_SIZE // 2, y + self.TILE_SIZE // 2),
                        self.TILE_SIZE // 3,
                    )
                    self.draw_text(
                        str(self.get_rock_weight(row_idx, col_idx)),
                        text_rect,
                        self.COLOR_BOX_WEIGHT,
                        self.COLOR_OCCUPIED,
                    )
                elif tile == ".":
                    pygame.draw.circle(
                        self.screen,
                        self.COLOR_GOAL,
                        (x + self.TILE_SIZE // 2, y + self.TILE_SIZE // 2),
                        self.TILE_SIZE // 3,
                    )

    def draw_text(self, text, rect: pygame.Rect, color, background):
        text_obj = self.font.render(text, True, color, background)
        text_rect = text_obj.get_rect()
        text_rect.topleft = (
            rect.left + rect.width // 2 - text_rect.width // 2,
            rect.top + rect.height // 2 - text_rect.height // 2,
        )
        self.screen.blit(text_obj, text_rect)

    def get_rock_weight(self, row, col):
        return self.rocks_map.get((row, col))

    def move_player(self, dx, dy, push=False):
        x, y = self.player_pos
        new_x, new_y = x + dx, y + dy
        target_tile = self.maze[new_x][new_y]

        if target_tile == " " or target_tile == ".":
            self.maze[new_x][new_y] = "@" if target_tile == " " else "+"
            self.maze[x][y] = " " if self.maze[x][y] == "@" else "."
            self.steps += 1
            self.player_pos = (new_x, new_y)
        elif target_tile == "$" or target_tile == "*":
            next_x, next_y = new_x + dx, new_y + dy
            next_tile = self.maze[next_x][next_y]

            if next_tile == " " or next_tile == ".":
                self.player_pos = (new_x, new_y)
                self.maze[next_x][next_y] = "$" if next_tile == " " else "*"
                self.maze[new_x][new_y] = "@" if target_tile == "$" else "+"
                self.maze[x][y] = " " if self.maze[x][y] == "@" else "."
                weight_index = self.get_rock_weight(new_x, new_y)
                if weight_index is not None:
                    self.cost += weight_index + 1
                self.steps += 1
                self.rocks_map[(next_x, next_y)] = self.rocks_map[(new_x, new_y)]
                del self.rocks_map[(new_x, new_y)]

        print(self.maze_to_string(self.maze))

    def maze_to_string(self, maze):
        s = ""
        for l in maze:
            for ch in l:
                s = s + ch
            s = s + "\n"
        return s

    def is_game_won(self):
        for x, y in self.goal_positions:
            if self.maze[x][y] != "$":
                return False
        return True

    def show_winning_screen(self):
        pass

    def draw_buttons(self):
        start_rect = pygame.draw.rect(
            self.screen, self.COLOR_BUTTON, (50, self.HEIGHT - 80, 100, 50)
        )
        pause_rect = pygame.draw.rect(
            self.screen, self.COLOR_BUTTON, (200, self.HEIGHT - 80, 100, 50)
        )
        reset_rect = pygame.draw.rect(
            self.screen, self.COLOR_BUTTON, (350, self.HEIGHT - 80, 100, 50)
        )
        self.draw_text("Start", start_rect, self.COLOR_TEXT, self.COLOR_BUTTON)
        self.draw_text("Pause", pause_rect, self.COLOR_TEXT, self.COLOR_BUTTON)
        self.draw_text("Reset", reset_rect, self.COLOR_TEXT, self.COLOR_BUTTON)

    def draw_cost(self):
        cost_text = self.font.render(
            f"Cost: {self.cost}", True, (255, 255, 255)
        )
        self.screen.blit(cost_text, (50, self.HEIGHT - 120))

    def draw_step_count(self):
        steps_count_text = self.font.render(
            f"Steps: {self.steps}", True, (255, 255, 255)
        )
        self.screen.blit(steps_count_text, (50, self.HEIGHT - 150))

    def handle_buttons(self, pos):
        if 50 <= pos[0] <= 150 and self.HEIGHT - 80 <= pos[1] <= self.HEIGHT - 30:
            self.running = True
            self.paused = False
        elif 200 <= pos[0] <= 300 and self.HEIGHT - 80 <= pos[1] <= self.HEIGHT - 30:
            self.paused = not self.paused
        elif 350 <= pos[0] <= 450 and self.HEIGHT - 80 <= pos[1] <= self.HEIGHT - 30:
            self.reset_game()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_buttons(event.pos)

            if self.running and not self.paused and self.move_index < len(self.moves):
                move = self.moves[self.move_index]
                if move == "u":
                    self.move_player(-1, 0)
                elif move == "d":
                    self.move_player(1, 0)
                elif move == "l":
                    self.move_player(0, -1)
                elif move == "r":
                    self.move_player(0, 1)
                elif move == "U":
                    self.move_player(-1, 0, push=True)
                elif move == "D":
                    self.move_player(1, 0, push=True)
                elif move == "L":
                    self.move_player(0, -1, push=True)
                elif move == "R":
                    self.move_player(0, 1, push=True)
                self.move_index += 1

            self.screen.fill(self.COLOR_BG)
            self.draw_maze()
            self.draw_buttons()
            self.draw_cost()
            self.draw_step_count()
            pygame.display.flip()
            self.clock.tick(self.FPS * self.speed)


class State:
    def __init__(self, maze, rock_weights, goals=None, player_pos=None, rocks_map=None):
        self.maze = maze
        self.rock_weights = rock_weights
        self.goals = self.find_goals() if goals is None else goals
        self.player_pos = self.find_player() if player_pos is None else player_pos
        self.rocks_map = self.find_rocks() if rocks_map is None else rocks_map

    def __str__(self):
        return str(self.maze)

    def find_goals(self):
        goals = []
        for row_idx, row in enumerate(self.maze):
            for col_idx, _ in enumerate(row):
                if self.maze[row_idx][col_idx] == ".":
                    goals.append((row_idx, col_idx))

        return goals

    def find_player(self):
        for row_idx, row in enumerate(self.maze):
            for col_idx, _ in enumerate(row):
                if self.maze[row_idx][col_idx] == "@":
                    return row_idx, col_idx

    def find_rocks(self):
        rocks_map = {}
        count = 0
        for row_idx, row in enumerate(self.maze):
            for col_idx, _ in enumerate(row):
                if (
                    self.maze[row_idx][col_idx] == "$"
                    or self.maze[row_idx][col_idx] == "*"
                ):
                    rocks_map[(row_idx, col_idx)] = self.rock_weights[count]
                    count += 1
        return rocks_map

    def __hash__(self):
        maze_hash = hash(tuple(tuple(row) for row in self.maze))
        rocks_map_hash = hash(frozenset(self.rocks_map.items()))
        return hash((maze_hash, rocks_map_hash))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.maze == other.maze and self.rocks_map == other.rocks_map
        return False

    def copy(self):
        return State(
            copy.deepcopy(self.maze),
            copy.deepcopy(self.rock_weights),
            copy.deepcopy(self.goals),
            copy.deepcopy(self.player_pos),
            copy.deepcopy(self.rocks_map),
        )


class Node:
    def __init__(self, state, parent, action, path_cost, weight_pushed, steps):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.weight_pushed = weight_pushed
        self.steps = steps

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.path_cost == other.path_cost

    def __hash__(self):
        return hash(self.state)


class Problem:
    actions = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}

    def __init__(self, initial: State):
        self.initial = initial

    def is_goal(self, state: State):
        for x, y in state.rocks_map.keys():
            if state.maze[x][y] == "$":
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
        new_state = state.copy()
        x, y = new_state.player_pos
        dx, dy = movement
        new_x, new_y = x + dx, y + dy
        dest_cell = new_state.maze[new_x][new_y]
        current_cell = new_state.maze[x][y]

        def update_cell(x, y, value):
            new_state.maze[x][y] = value

        # Move to empty cell
        if dest_cell in (" ", "."):
            new_state.player_pos = (new_x, new_y)
            update_cell(new_x, new_y, "@" if dest_cell == " " else "+")
            update_cell(x, y, " " if current_cell == "@" else ".")
            return new_state, False, 1
        # Move to box
        elif dest_cell in ("$", "*"):
            next_x, next_y = new_x + dx, new_y + dy
            next_cell = new_state.maze[next_x][next_y]
            if next_cell in (" ", "."):
                new_state.player_pos = (new_x, new_y)
                update_cell(next_x, next_y, "$" if next_cell == " " else "*")
                update_cell(new_x, new_y, "@" if dest_cell == "$" else "+")
                update_cell(x, y, " " if current_cell == "@" else ".")
                new_state.rocks_map[(next_x, next_y)] = new_state.rocks_map[
                    (new_x, new_y)
                ]
                del new_state.rocks_map[(new_x, new_y)]
                return new_state, True, new_state.rocks_map[(next_x, next_y)] + 1

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
        print(self.start_time)

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

        time_taken = (self.end_time - self.start_time) * 1000  # Convert to milliseconds
        memory_used = (self.memory_end - self.memory_start) / (
            1024 * 1024
        )  # Convert to MB
        return f"{self.algorithm_name}\nSteps: {self.steps}, Cost: {self.total_cost}, Node: {self.nodes_generated}, Time (ms): {time_taken:.2f}, Memory (MB): {memory_used:.2f}\n{self.result}"


class AStarSolver(Solver):
    def __init__(self):
        super().__init__("A*")

    def solve(self, problem: Problem):
        super().solve(problem)

        node = Node(self.problem.initial, None, None, 0, 0, 0)
        frontier = PriorityQueue()
        reached = {}
        heuristic = {}
        self.nodes_generated = 1

        frontier.put(node)  # cost, node
        reached[node.state] = 0  # cost of reaching the node
        heuristic[node.state] = self.heuristic_cost(node.state)

        while not frontier.empty():
            node = frontier.get()
            state = node.state

            if self.problem.is_goal(state):
                return self.trace_path(node)

            for action in self.problem.actions:
                child_state, box_moved, moving_cost = self.problem.result(
                    node.state, action
                )
                if child_state is None:
                    continue

                is_child_reached_before = reached.get(child_state) is not None
                if not is_child_reached_before:
                    heuristic[child_state] = self.heuristic_cost(child_state)

                child_cost = node.path_cost + moving_cost + heuristic[child_state]

                if not is_child_reached_before or reached[child_state] > child_cost:
                    reached[child_state] = child_cost

                    real_action = action
                    if box_moved:
                        real_action = action.upper()

                    child_node = Node(
                        child_state,
                        node,
                        real_action,
                        child_cost,
                        node.weight_pushed + moving_cost - 1,
                        node.steps + 1,
                    )
                    frontier.put(child_node)
                    self.nodes_generated += 1

        return None

    def heuristic_cost(self, state):
        return 0  # UCS

    def trace_path(self, node):
        self.steps = node.steps
        self.total_weight_pushed = node.weight_pushed
        self.total_cost = node.path_cost
        path = []
        while node.parent:
            path.append(node.action)
            node = node.parent

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
    # Hint: You can use the `queue.Queue` class to implement the frontier
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


class App:
    BUTTON_WIDTH = 120
    BUTTON_HEIGHT = 60
    DROPDOWN_WIDTH = 120
    DROPDOWN_HEIGHT = 60
    INACTIVE_COLOR = (0, 255, 255)
    PRESSED_COLOR = (255, 0, 255)
    HOVER_COLOR = (255, 255, 0)

    def __init__(self):
        self.io_handler = IOHandler()
        self.solvers = {
            "A*": AStarSolver(),
            "DFS": DFSolver(),
            "BFS": BFSolver(),
            "UCS": UCSolver(),
        }
        self.metric_results = {}
        self.algorithm_results = {}
        self.maps = {}
        self.visualizer = Visualizer()
        self.current_map_name = None
        self.current_algorithm_name = None

        self.prepare_maps()
        self.prepare_ui()

    def prepare_maps(self):
        # Get all maps file with the prefix "input" in the folder
        # Suppose the file name is input-01.txt, input-02.txt, etc.
        for file in os.listdir():
            if file.startswith("input") and file.endswith(".txt"):
                self.io_handler.set_input_file_name(file)
                self.maps[file] = self.io_handler.parse()

    def prepare_ui(self):
        pygame.init()
        pygame.display.set_caption("Ares's Adventure")

        # Get full screen size and set to half
        info = pygame.display.Info()
        self.WIDTH = info.current_w // 1
        self.HEIGHT = info.current_h // 1

        self.font = pygame.font.Font("freesansbold.ttf", 18)
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((800, 600))
        self.map_dropdown = Dropdown(
            self.window,
            10,
            10,
            self.DROPDOWN_WIDTH,
            self.DROPDOWN_HEIGHT,
            name="Map",
            choices=[key for key in self.maps.keys()],
            borderRadius=20,
            fontSize=20,
            inactiveColour=(0, 255, 255),
            pressedColour=(255, 0, 255),
            hoverColour=(255, 255, 0),
        )
        self.algorithm_dropdown = Dropdown(
            self.window,
            self.map_dropdown._x + self.DROPDOWN_WIDTH + 10,
            self.map_dropdown._y,
            self.DROPDOWN_WIDTH,
            self.DROPDOWN_HEIGHT,
            name="Algorithm",
            choices=["A*", "DFS", "BFS", "UCS"],
            borderRadius=20,
            fontSize=20,
            inactiveColour=(0, 255, 255),
            pressedColour=(255, 0, 255),
            hoverColour=(255, 255, 0),
        )
        self.visualize_button = Button(
            self.window,
            self.algorithm_dropdown._x + self.DROPDOWN_WIDTH + 10,
            self.algorithm_dropdown._y,
            self.BUTTON_WIDTH,
            self.BUTTON_HEIGHT,
            text="Visualize",
            fontSize=20,
            margin=20,
            inactiveColour=(0, 255, 255),
            pressedColour=(255, 0, 255),
            hoverColour=(255, 255, 0),
            radius=20,
            onClick=lambda: self.visualize(
                self.map_dropdown.getSelected(), self.algorithm_dropdown.getSelected()
            ),
        )
        self.start_button = Button(
            self.window,
            10,
            self.HEIGHT - 10,
            self.BUTTON_WIDTH,
            self.BUTTON_HEIGHT,
            text="Start",
            fontSize=20,
            margin=20,
            inactiveColour=(0, 255, 255),
            pressedColour=(255, 0, 255),
            hoverColour=(255, 255, 0),
            radius=20,
        )
        self.pause_button = Button(
            self.window,
            self.start_button._x + self.BUTTON_WIDTH + 10,
            self.start_button._y,
            self.BUTTON_WIDTH,
            self.BUTTON_HEIGHT,
            text="Pause",
            fontSize=20,
            margin=20,
            inactiveColour=(0, 255, 255),
            pressedColour=(255, 0, 255),
            hoverColour=(255, 255, 0),
            radius=20,
        )
        self.reset_button = Button(
            self.window,
            self.pause_button._x + self.BUTTON_WIDTH + 10,
            self.pause_button._y,
            self.BUTTON_WIDTH,
            self.BUTTON_HEIGHT,
            text="Reset",
            fontSize=20,
            margin=20,
            inactiveColour=(0, 255, 255),
            pressedColour=(255, 0, 255),
            hoverColour=(255, 255, 0),
            radius=20,
        )
        self.speed_slider = Slider(
            self.window,
            600,
            0,
            200,
            60,
            min=0,
            max=1,
            step=0.1,
            initial=0.5,
            handleRadius=20,
            handleColour=(0, 255, 255),
            handleOutline=(255, 0, 255),
            handleSize=20,
            barColour=(255, 255, 0),
            barOutline=(0, 0, 0),
            barSize=20,
            onSlide=lambda: self.update_speed(self.spped_slider.getValue()),
        )

    def visualize(self, map_name, algorithm_name):
        self.choose_map(map_name)
        self.choose_algorithm(algorithm_name)
        if self.current_map_name is None or self.current_algorithm_name is None:
            print("Please choose a map and an algorithm first!")
            return

        self.run_solvers(self.current_map_name)
        if self.algorithm_results[self.current_algorithm_name] is None:
            print("No solution found!")
            return
        self.visualizer.change_map(
            self.maps[self.current_map_name][0],
            self.maps[self.current_map_name][1],
            self.algorithm_results[self.current_algorithm_name],
        )
        self.visualizer.run()

    def run_solvers(self, map_name):
        maze, rock_weights = self.maps[map_name]
        initial_state = State(maze, rock_weights)
        problem = Problem(initial_state)

        for solver in self.solvers.values():
            print(type(solver))
            solver.change_problem(problem)
            self.algorithm_results[solver.algorithm_name] = solver.solve_and_measure(
                problem
            )
            self.metric_results[solver.algorithm_name] = solver.output_metrics()

        print(self.metric_results)
        self.io_handler.write_metrics_result("\n".join(self.metric_results.values()))

    def preview_map(self, map_name):
        pass

    def choose_map(self, map_name):
        self.current_map_name = map_name
        self.preview_map(map_name)
        self.io_handler.set_input_file_name(map_name)
        print(f"Map {map_name} is chosen.")
        pass

    def choose_algorithm(self, algorithm_name):
        self.current_algorithm_name = algorithm_name
        print(f"Algorithm {algorithm_name} is chosen.")
        pass

    def run(self):
        run = True
        while run:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    run = False
                    quit()

            self.window.fill((255, 255, 255))

            pygame_widgets.update(events)
            pygame.display.update()


def main():
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    app = App()
    app.run()


if __name__ == "__main__":
    main()
