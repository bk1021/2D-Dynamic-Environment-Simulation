import numpy as np
import random
import matplotlib.pyplot as plt
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

    def add_polygon_obstacle(self, vertices):
        """
        Add a polygonal obstacle to the environment.
        
        :param vertices: List of (x, y) coordinates defining the polygon
        """
        # Create a Shapely Polygon from the vertices
        polygon = Polygon(vertices)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)
            if isinstance(polygon, MultiPolygon):
                polygon = next(iter(polygon.geoms))
        
        # Add the obstacle to the list
        self.obstacles.append(polygon)
        
        # Mark the grid cells as obstacles (0) within the polygon
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
            # Randomly generate the number of vertices for the polygon
            num_vertices = random.randint(3, max_vertices)

            # Randomly generate a center point for the polygon
            center_x = random.randint(0, self.width - 1)
            center_y = random.randint(0, self.height - 1)

            # Generate vertices around the center point
            vertices = []
            for _ in range(num_vertices):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(min_size, max_size)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                # Ensure vertices are within bounds
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
            return True
        return False

    def visualize(self):
        """
        Visualize the environment with obstacles and robot's position.
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Plot the obstacles
        for obs in self.obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='r', label="Obstacle")
        
        # Plot the robot's current position
        ax.plot(self.robot_position[0], self.robot_position[1], 'bo', label="Robot")
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title('2D Simulated Environment')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        # ax.grid(True)
        # ax.legend()
        
        plt.show()


# Example Usage
if __name__ == '__main__':
    # Create a 20x20 environment
    env = Simulated2DEnvironment(20, 20)
    
    # Add polygon obstacles
    env.add_polygon_obstacle([(3, 3), (7, 3), (7, 7), (3, 7)])  # Square obstacle
    env.add_polygon_obstacle([(12, 12), (16, 12), (16, 16), (12, 16)])  # Another square

    env.generate_random_obstacles(2, random_seed=87)
    
    # Visualize the environment
    env.visualize()

    # Simulate robot movement
    print("Initial Position:", env.robot_position)
    env.move_robot(1, 0)  # Move right
    env.move_robot(0, 1)  # Move down
    
    # Check new robot position
    # print("New Position:", env.robot_position)
    
    # # Visualize after movement
    # env.visualize()
