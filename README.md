# Roomba Path Planning Simulation
Use matplotlib to simulate a 2D environment with static obstacles, the simulated environment can be used to test robotics path planning algorithms.

## Class
#### `Environment`
#### Argument
`size_x`, `size_y`: width, height of the environment

`n_obs`: number of obstacles (optional)

`resolution`: resolution of the configuration space (optional)

`random_seed`: to replicate result (optional)
#### Attributes
`self.size_x`, `self.size_y`, `self.resolution`

`self.obs`: A list storing obstacles as `Polygon` object

`self.cspace`: A 2D np.array of x,y (row, col) configuration space, True for free-collision, False otherwise

`self.x_grid`, `self.y_grid`: Store x, y values for configuration space
#### Method
`generate_cspace(self, resolution=0.1)`:

`check_collision(self, point, buffer=0.1)`:

`check_line_collision(self, p1, p2, step=100, buffer=0.1)`:

`add_polyobs(self, vertices)`:

`generate_obstacles(self, n_obs, max_vertices=6)`:

`generate_start_goal(self)`:

`xy_to_idx(self, xy)`:

`is_valid_move(self, idx)`:

`ifReachGoal(self, xy)`:

`evaluate_performance(self, sol_path, visited_idx, output_file='performance_metrics.txt')`:

`visualize(self, sol_path)`: 

## Usage
```
gg
```

