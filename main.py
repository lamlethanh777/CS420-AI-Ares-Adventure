import pygame




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

    def __init__(self, maze, moves, rock_weights):
        self.original_maze = maze
        self.moves = moves
        self.rock_weights = rock_weights
        self.reset_game()

        pygame.init()
        pygame.display.set_caption("Ares's Adventure")

        # Get full screen size and set to half
        info = pygame.display.Info()
        self.WIDTH = info.current_w // 2
        self.HEIGHT = info.current_h // 2

        # Calculate new tile size
        self.TILE_SIZE = min(self.WIDTH // len(maze[0]), self.HEIGHT // len(maze))

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.Font("freesansbold.ttf", 18)
        self.clock = pygame.time.Clock()

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
        self.weight_pushed = 0
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
                    self.weight_pushed += weight_index
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
        # start_text = self.font.render("Start", True, (255, 255, 255))
        # pause_text = self.font.render("Pause", True, (255, 255, 255))
        # reset_text = self.font.render("Reset", True, (255, 255, 255))
        # self.screen.blit(start_text, (75, self.HEIGHT - 70))
        # self.screen.blit(pause_text, (225, self.HEIGHT - 70))
        # self.screen.blit(reset_text, (375, self.HEIGHT - 70))
        self.draw_text("Start", start_rect, self.COLOR_TEXT, self.COLOR_BUTTON)
        self.draw_text("Pause", pause_rect, self.COLOR_TEXT, self.COLOR_BUTTON)
        self.draw_text("Reset", reset_rect, self.COLOR_TEXT, self.COLOR_BUTTON)

    def draw_weight_pushed(self):
        weight_pushed_text = self.font.render(f"Weight pushed: {self.weight_pushed}", True, (255, 255, 255))
        self.screen.blit(weight_pushed_text, (50, self.HEIGHT - 120))

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
            self.draw_weight_pushed()
            self.draw_step_count()
            pygame.display.flip()
            self.clock.tick(self.FPS * self.speed)

# Example usage
maze = [
    " ###########",
    "##         #",
    "#          #",
    "# $ $      #",
    "#. @      .#",
    "############",
]

moves = "uLulDrrRRRRRRurD"
rock_weights = [1, 99]

visualizer = Visualizer(maze, moves, rock_weights)
visualizer.run()
