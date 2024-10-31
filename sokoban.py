import tkinter as tk
from tkinter import ttk

class SokobanSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sokoban Simulator")
        
        # Dropdowns for map and algorithm selection
        self.map_var = tk.StringVar()
        self.algorithm_var = tk.StringVar()
        
        self.map_dropdown = ttk.Combobox(self, textvariable=self.map_var, state='readonly')
        self.map_dropdown['values'] = ('Map1', 'Map2')  # Placeholder maps
        self.map_dropdown.current(0)
        self.map_dropdown.grid(row=0, column=0, padx=10, pady=10)
        
        self.algorithm_dropdown = ttk.Combobox(self, textvariable=self.algorithm_var, state='readonly')
        self.algorithm_dropdown['values'] = ('Algorithm1',)  # Placeholder algorithm
        self.algorithm_dropdown.current(0)
        self.algorithm_dropdown.grid(row=0, column=1, padx=10, pady=10)
        
        # Button to start computation
        self.compute_button = tk.Button(self, text="Compute Path", command=self.compute_path)
        self.compute_button.grid(row=0, column=2, padx=10, pady=10)
        
        # Buttons for start, pause, and reset
        self.start_button = tk.Button(self, text="Start", command=self.start_simulation, state='disabled')
        self.start_button.grid(row=2, column=0, padx=10, pady=10)
        
        self.pause_button = tk.Button(self, text="Pause", command=self.pause_simulation, state='disabled')
        self.pause_button.grid(row=2, column=1, padx=10, pady=10)
        
        self.reset_button = tk.Button(self, text="Reset", command=self.reset_simulation, state='disabled')
        self.reset_button.grid(row=2, column=2, padx=10, pady=10)
        
        # Slider for simulation speed
        self.speed_slider = tk.Scale(self, from_=1, to=10, orient=tk.HORIZONTAL, label="Speed")
        self.speed_slider.set(5)
        self.speed_slider.grid(row=2, column=3, padx=10, pady=10)
        
        # Canvas for the game screen
        self.canvas = tk.Canvas(self, width=600, height=600, bg="white")
        self.canvas.grid(row=1, column=0, columnspan=4, padx=10, pady=10)
        
        # Labels for number of steps and weight of rocks pushed
        self.steps_label = tk.Label(self, text="Steps: 0")
        self.steps_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='w')
        
        self.weight_label = tk.Label(self, text="Weight: 0")
        self.weight_label.grid(row=3, column=2, columnspan=2, padx=10, pady=10, sticky='e')
        
        # Initialize game variables
        self.steps = 0
        self.weight = 0
        self.map = []
        self.rock_weights = []
        self.moves = ""
        self.current_move_index = 0
        self.simulation_running = False
        self.simulation_paused = False
        
        self.cell_size = 0
        self.player_position = (0, 0)
        self.rock_positions = {}
        self.goal_positions = set()
        
        # Sample maps
        self.maps = {
            'Map1': {
                'map_data': [
                    list(" ##### "),
                    list(" #   # "),
                    list(" #$@.# "),
                    list(" ##### ")
                ],
                'rock_weights': [2],
                'moves': "RRUULLDD"  # Placeholder moves
            },
            'Map2': {
                'map_data': [
                    list("  #### "),
                    list("###  # "),
                    list("#.$@ # "),
                    list("###  # "),
                    list("  #### ")
                ],
                'rock_weights': [3],
                'moves': "UURRDDLL"  # Placeholder moves
            }
        }
        
    def compute_path(self):
        map_name = self.map_var.get()
        algorithm_name = self.algorithm_var.get()
        
        # Load the selected map
        self.load_map(map_name)
        
        # Get the moves (placeholder for actual algorithm)
        self.moves = self.maps[map_name]['moves']
        
        # Enable the start, pause, and reset buttons
        self.start_button.config(state='normal')
        self.pause_button.config(state='normal')
        self.reset_button.config(state='normal')
        
        # Reset simulation variables
        self.simulation_running = False
        self.simulation_paused = False
        self.steps = 0
        self.weight = 0
        self.steps_label.config(text="Steps: 0")
        self.weight_label.config(text="Weight: 0")
        self.current_move_index = 0
        
        # Draw the initial map
        self.draw_map()
        
    def load_map(self, map_name):
        map_data = self.maps[map_name]['map_data']
        self.map = [row[:] for row in map_data]
        self.rock_weights = self.maps[map_name]['rock_weights'][:]
        self.player_position = None
        self.rock_positions = {}
        self.goal_positions = set()
        
        rock_index = 0
        for y, row in enumerate(self.map):
            for x, cell in enumerate(row):
                if cell == '@' or cell == '+':
                    self.player_position = (x, y)
                if cell == '$' or cell == '*':
                    self.rock_positions[(x, y)] = self.rock_weights[rock_index]
                    rock_index += 1
                if cell == '.' or cell == '*' or cell == '+':
                    self.goal_positions.add((x, y))
        
        # Calculate cell size
        map_width = max(len(row) for row in self.map)
        map_height = len(self.map)
        self.cell_size = min(600 // map_width, 600 // map_height)
        self.canvas.config(width=map_width * self.cell_size, height=map_height * self.cell_size)
        
    def start_simulation(self):
        if not self.simulation_running:
            self.simulation_running = True
            self.simulation_paused = False
            self.update_simulation()
        else:
            self.simulation_paused = False  # Resume if paused
    
    def pause_simulation(self):
        self.simulation_paused = True
    
    def reset_simulation(self):
        self.simulation_running = False
        self.simulation_paused = False
        self.current_move_index = 0
        self.steps = 0
        self.weight = 0
        self.steps_label.config(text="Steps: 0")
        self.weight_label.config(text="Weight: 0")
        self.load_map(self.map_var.get())
        self.draw_map()
    
    def update_simulation(self):
        if self.simulation_running and not self.simulation_paused:
            if self.current_move_index < len(self.moves):
                move = self.moves[self.current_move_index]
                self.current_move_index += 1
                self.make_move(move)
                self.steps += 1
                self.steps_label.config(text=f"Steps: {self.steps}")
                delay = int(1000 / self.speed_slider.get())
                self.after(delay, self.update_simulation)
            else:
                self.simulation_running = False
    
    def make_move(self, move):
        dx, dy = 0, 0
        if move == 'U':
            dx, dy = 0, -1
        elif move == 'D':
            dx, dy = 0, 1
        elif move == 'L':
            dx, dy = -1, 0
        elif move == 'R':
            dx, dy = 1, 0
        else:
            return
        
        x, y = self.player_position
        new_x, new_y = x + dx, y + dy
        
        # Check bounds
        if not (0 <= new_y < len(self.map) and 0 <= new_x < len(self.map[0])):
            return
        
        target_cell = self.map[new_y][new_x]
        
        if target_cell in (' ', '.'):
            # Move player
            self.update_map_cell(x, y, ' ' if self.map[y][x] == '@' else '.')
            self.player_position = (new_x, new_y)
            self.update_map_cell(new_x, new_y, '@' if target_cell == ' ' else '+')
        elif target_cell in ('$','*'):
            rock_x, rock_y = new_x + dx, new_y + dy
            if 0 <= rock_y < len(self.map) and 0 <= rock_x < len(self.map[0]):
                next_cell = self.map[rock_y][rock_x]
                if next_cell in (' ', '.'):
                    # Move rock
                    self.update_map_cell(rock_x, rock_y, '$' if next_cell == ' ' else '*')
                    self.update_map_cell(new_x, new_y, '@' if target_cell == '$' else '+')
                    self.update_map_cell(x, y, ' ' if self.map[y][x] == '@' else '.')
                    # Update positions
                    weight = self.rock_positions.pop((new_x, new_y))
                    self.rock_positions[(rock_x, rock_y)] = weight
                    self.player_position = (new_x, new_y)
                    # Update weight
                    self.weight += weight
                    self.weight_label.config(text=f"Weight: {self.weight}")
        self.draw_map()
    
    def update_map_cell(self, x, y, value):
        self.map[y][x] = value
    
    def draw_map(self):
        self.canvas.delete("all")
        for y, row in enumerate(self.map):
            for x, cell in enumerate(row):
                x1 = x * self.cell_size
                y1 = y * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                if cell == '#':
                    color = 'black'
                elif cell == ' ':
                    color = 'white'
                elif cell == '@':
                    color = 'blue'
                elif cell == '$':
                    color = 'brown'
                elif cell == '.':
                    color = 'yellow'
                elif cell == '+':
                    color = 'green'
                elif cell == '*':
                    color = 'red'
                else:
                    color = 'white'
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='grey')
                if cell in ('$','*'):
                    weight = self.rock_positions.get((x, y), 0)
                    self.canvas.create_text(x1 + self.cell_size // 2, y1 + self.cell_size // 2, text=str(weight), fill='white')
    
if __name__ == "__main__":
    app = SokobanSimulator()
    app.mainloop()
