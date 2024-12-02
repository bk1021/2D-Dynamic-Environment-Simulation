import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, Point, MultiPolygon


class Simulated2DEnvironment:
    def __init__(self, width, height):
        """
        Initialize the 2D environment.
        
        :param width: Width of the environment
        :param height: Height of the environment
        """
        self.width = width
        self.height = height
        self.obstacles = []
        self.robot_position = np.array([width // 2, height // 2])  # Start at the center
        self.grid = np.ones((height, width))  # 1 indicates free space, 0 indicates obstacles
        self.visited_cells = np.zeros((height, width), dtype=int)  # Track visited cells

    def add_polygon_obstacle(self, vertices):
        """
        Add a polygonal obstacle to the environment.
        
        :param vertices: List of (x, y) coordinates defining the polygon
        """
        polygon = Polygon(vertices)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)
            if isinstance(polygon, MultiPolygon):
                polygon = next(iter(polygon.geoms))
        
        self.obstacles.append(polygon)
        
        for i in range(self.height):
            for j in range(self.width):
                point = Point(j, i)
                if polygon.contains(point):
                    self.grid[i, j] = 0  # Mark as obstacle

    def generate_random_obstacles(self, num_obstacles, max_vertices=6, min_size=3, max_size=7, random_seed=None):
        """
        Generate a random number of random polygon obstacles.

        :param num_obstacles: Number of obstacles to generate
        :param max_vertices: Maximum number of vertices per polygon
        :param min_size: Minimum size of the polygon (distance between vertices)
        :param max_size: Maximum size of the polygon (distance between vertices)
        """
        if random_seed is not None:
            random.seed(random_seed)

        for _ in range(num_obstacles):
            num_vertices = random.randint(3, max_vertices)
            center_x = random.randint(0, self.width - 1)
            center_y = random.randint(0, self.height - 1)

            vertices = []
            for _ in range(num_vertices):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(min_size, max_size)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                x = max(0, min(self.width - 1, x))
                y = max(0, min(self.height - 1, y))
                vertices.append((x, y))

            self.add_polygon_obstacle(vertices)

    def is_valid_move(self, x, y):
        """
        Check if a given position (x, y) is valid (i.e., not within an obstacle).
        
        :param x: x-coordinate
        :param y: y-coordinate
        :return: True if valid, False if blocked by an obstacle
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False  # Out of bounds
        return self.grid[y, x] == 1  # Check if free space

    def move_robot(self, dx, dy):
        """
        Move the robot by (dx, dy) if the move is valid.
        
        :param dx: Change in x-coordinate
        :param dy: Change in y-coordinate
        :return: True if moved, False if blocked
        """
        new_x = self.robot_position[0] + dx
        new_y = self.robot_position[1] + dy
        if self.is_valid_move(new_x, new_y):
            self.robot_position = np.array([new_x, new_y])
            self.visited_cells[new_y, new_x] = 1  # Mark cell as visited
            return True
        return False

    def get_neighbors(self, pos):
        """Get valid neighboring positions."""
        x, y = pos
        neighbors = [
            (x + dx, y + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if self.is_valid_move(x + dx, y + dy)
        ]
        return neighbors

    def boustrophedon_traversal(self):
        """
        Perform boustrophedon traversal to cover the entire area.
        """
        visited_positions = set()
        stack = [tuple(self.robot_position)]  # Convert to tuple for hashability
        direction = 1  # 1 for right/down, -1 for left/up
        path = []  # Store the path for animation

        while stack:
            current_pos = stack.pop()
            if current_pos in visited_positions:
                continue
            visited_positions.add(current_pos)
            self.robot_position = np.array(current_pos)
            path.append(self.robot_position.copy())  # Record each step

            neighbors = sorted(
                self.get_neighbors(current_pos),
                key=lambda p: p[0] if direction == 1 else -p[0]
            )

            if not neighbors:
                direction *= -1
                potential_moves = [(current_pos[0], current_pos[1] + direction), (current_pos[0], current_pos[1] - direction)]
                for move in potential_moves:
                    if self.is_valid_move(*move) and move not in visited_positions:
                        stack.append(move)
                        break
            else:
                stack.extend(neighbors)

        print("Coverage completed.")
        return path

    def visualize(self, fig, ax, path):
        """
        Visualize the environment with obstacles, robot's position, and visited cells.
        """
        ax.clear()

        # Plot the grid with obstacles and visited cells
        ax.imshow(self.grid, cmap='Greys', origin='lower')
        ax.imshow(self.visited_cells, cmap='Blues', alpha=0.5, origin='lower')

        # Plot the obstacles
        for obs in self.obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='r', ec='k', label="Obstacle" if len(self.obstacles) == 1 else "")

        # Plot the robot's path
        path_x, path_y = zip(*[(pos[0], pos[1]) for pos in path])
        ax.plot(path_x, path_y, 'r-', label="Robot Path")
        ax.plot(self.robot_position[0], self.robot_position[1], 'ro', label="Robot")

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title('2D Simulated Environment')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend(loc='upper right')
        plt.tight_layout()

# Example Usage
if __name__ == '__main__':
    # Create a 20x20 environment
    env = Simulated2DEnvironment(20, 20)
    
    # Add polygon obstacles
    env.add_polygon_obstacle([(3, 3), (7, 3), (7, 7), (3, 7)])  # Square obstacle
    env.add_polygon_obstacle([(12, 12), (16, 12), (16, 16), (12, 16)])  # Another square

    env.generate_random_obstacles(2, random_seed=87)
    
    # Perform boustrophedon traversal and get the path
    path = env.boustrophedon_traversal()

    # Set up the figure and axis for animation
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        env.visualize(fig, ax, path[:frame+1])

    ani = FuncAnimation(fig, update, frames=len(path), interval=100, repeat=False)

    plt.show()