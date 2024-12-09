import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, box, GeometryCollection, Point
from shapely.ops import unary_union
from simulation import Environment, bresenham_line
from collections import defaultdict
import heapq

class VerticalStripDecomposition:
    def __init__(self, env):
        """
        Initialize the workspace and obstacles.

        Args:
        - workspace_bounds: Tuple (min_x, min_y, max_x, max_y).
        - obstacles: List of Shapely Polygons representing obstacles.
        """
        self.env = env
        self.cspace = self.env.cspace
        workspace_bounds = (0, 0, self.env.size_x, self.env.size_y)
        self.workspace = box(*workspace_bounds)
        self.obstacles = self.env.obs
        self.free_space = self.workspace.difference(self._merge_obstacles())

    def _merge_obstacles(self):
        """
        Merge all obstacles into a single Shapely geometry.
        """
        return unary_union(self.obstacles)   

    def decompose(self):
        """
        Decompose the workspace into vertical strips based on obstacle boundaries
        and merge strips if they meet the conditions, regardless of adjacency.
        """
        # Step 1: Determine boundaries for vertical strips
        boundaries = []
        for obs in self.obstacles:
            min_x, _, max_x, _ = obs.bounds
            boundaries.append(min_x)
            boundaries.append(max_x)

        # Sort boundaries and include workspace edges
        boundaries = sorted(set(boundaries + [self.workspace.bounds[0], self.workspace.bounds[2]]))

        # Step 2: Create initial strips
        strips = []
        for i in range(len(boundaries) - 1):
            strip = box(boundaries[i], self.workspace.bounds[1], boundaries[i + 1], self.workspace.bounds[3])
            free_strip = strip.intersection(self.free_space)

            # Handle different types of geometry outputs
            if isinstance(free_strip, MultiPolygon):
                strips.extend(free_strip.geoms)
            elif isinstance(free_strip, Polygon):
                strips.append(free_strip)
            elif isinstance(free_strip, GeometryCollection):
                for geom in free_strip.geoms:
                    if isinstance(geom, Polygon):
                        strips.append(geom)

        # Function to collect mergeable strips
        def collect_mergeable_strips(strips):
            merge_groups = []
            visited = set()

            for i, strip_a in enumerate(strips):
                if i in visited:
                    continue
                group = [strip_a]
                for j, strip_b in enumerate(strips):
                    if j == i or j in visited:
                        continue
                    # Check if both strips intersect the same single obstacle
                    intersecting_obstacles_a = [obs for obs in self.obstacles if strip_a.intersects(obs)]
                    intersecting_obstacles_b = [obs for obs in self.obstacles if strip_b.intersects(obs)]

                    if (
                        len(intersecting_obstacles_a) == 1
                        and len(intersecting_obstacles_b) == 1
                        and intersecting_obstacles_a[0] == intersecting_obstacles_b[0]
                    ):
                        # Check relative position
                        obstacle = intersecting_obstacles_a[0]
                        if (
                            strip_a.bounds[3] <= obstacle.bounds[3] and strip_b.bounds[3] <= obstacle.bounds[3]
                            or strip_a.bounds[1] >= obstacle.bounds[1] and strip_b.bounds[1] >= obstacle.bounds[1]
                        ):
                            group.append(strip_b)
                            visited.add(j)
                    elif (
                        len(intersecting_obstacles_a) == 2
                        and len(intersecting_obstacles_b) == 2
                        and (intersecting_obstacles_a == intersecting_obstacles_b or intersecting_obstacles_a == reversed(intersecting_obstacles_b))
                    ):
                        # Check relative position
                        obstacle1 = intersecting_obstacles_a[0]
                        obstacle2 = intersecting_obstacles_a[1]
                        if obstacle1.bounds[3] > obstacle2.bounds[3]:
                            upper_obs, lower_obs = obstacle1, obstacle2
                        else:
                            upper_obs, lower_obs = obstacle2, obstacle1
                        if (
                            strip_a.bounds[1] >= lower_obs.bounds[1] and strip_b.bounds[1] >= lower_obs.bounds[1]
                            and strip_a.bounds[3] <= upper_obs.bounds[3] and strip_b.bounds[3] <= upper_obs.bounds[3]
                        ):
                            group.append(strip_b)
                            visited.add(j)

                merge_groups.append(group)

            return merge_groups

        # Function to merge strips
        def merge_strips(strips):
            merge_groups = collect_mergeable_strips(strips)
            merged_strips = []

            for group in merge_groups:
                if len(group) > 1:
                    # Sort by min_x before merging
                    group = sorted(group, key=lambda s: s.bounds[0])
                    merged_strips.append(unary_union(group))
                else:
                    merged_strips.extend(group)

            return merged_strips
        
        merged_strips = merge_strips(strips)

        return merged_strips


    def plan_motion_within_strip(self, strip):
        strip_cells = []
        waypoints = []
        for i in range(self.cspace.shape[0]):
            for j in range(self.cspace.shape[1]):
                if self.cspace[i, j]:
                    x, y = self.env.x_grid[i, j], self.env.y_grid[i, j]
                    point = Point((x, y))
                    if strip.intersects(point):
                        strip_cells.append((x, y))

        # Sort strip_cells by x value
        strip_cells.sort(key=lambda cell: cell[0])

        # Group by x_value and determine y_min and y_max
        x_groups = defaultdict(list)

        for x, y in strip_cells:
            x_groups[x].append(y)

        # Determine y bounds for each x value and maintain sorted order
        x_y_bounds = {}
        sorted_x_values = sorted(x_groups.keys())

        for x in sorted_x_values:
            y_values = x_groups[x]
            y_min = min(y_values)
            y_max = max(y_values)
            x_y_bounds[x] = (y_min, y_max)

        direction = 1 # 1 for downward, -1 for upward
        for x in sorted_x_values:
            y_min, y_max = x_y_bounds[x]
            if direction == 1:
                waypoints.append((x, y_min))
                waypoints.append((x, y_max))
            else:
                waypoints.append((x, y_max))
                waypoints.append((x, y_min))

            direction *= -1

        return waypoints
    

    def combined_strips_waypoints(self, strips):

        combined_waypoints = []
        for strip in strips:
            combined_waypoints.extend(self.plan_motion_within_strip(strip))
        
        return combined_waypoints
    

    def waypoints_to_path(self, waypoints):
        sol_path = [waypoints[0]]
        for i in range(len(waypoints) - 1):
            path = self.a_star(waypoints[i], waypoints[i+1])
            if path is not None:
                sol_path.extend(path)
            else:
                print("The waypoints have no solution!")
                return
        return sol_path
    

    def boustrophedon_path_planning(self):
        strips = self.decompose()
        waypoints = self.combined_strips_waypoints(strips)
        sol_path = self.waypoints_to_path(waypoints)

        return sol_path, strips, waypoints


    def expand_node(self, current, speed_limit=(1, 1)):
        """Generate free-collision neighboring configurations within speed limit"""
        resolution = self.env.resolution
        cspace = self.cspace
        idx = self.env.xy_to_idx(current)

        def is_valid_neighbor(current_idx, neighbor_idx):
            idx_list = bresenham_line(current_idx[0], current_idx[1], neighbor_idx[0], neighbor_idx[1])
            for idx in idx_list:
                if not self.env.is_valid_move(idx):
                    return False
            return True

        neighbors = []
        max_dr = int(abs(speed_limit[0]/resolution))
        max_dc = int(abs(speed_limit[1]/resolution))

        # Define the relative positions of the all neighbors
        neighbor_offsets = [(dr,dc) for dr in range(-max_dr, max_dr+1) for dc in range(-max_dc, max_dc+1) if (dr,dc) != (0,0)]

        for dr, dc in neighbor_offsets:
            new_row = idx[0] + dr
            new_col = idx[1] + dc
            
            # Check if the neighbor is within the cspace and not an obstacle
            if 0 <= new_row < cspace.shape[0] and 0 <= new_col < cspace.shape[1]:
                if is_valid_neighbor(idx, (new_row, new_col)):
                    neighbor = self.env.idx_to_xy((new_row, new_col))
                    neighbors.append(neighbor)

        return neighbors


    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))    


    def a_star(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {(x, y): float('inf') for x in self.env.x_grid[:, 0] for y in self.env.y_grid[0, :]}
        g_score[start] = 0
        f_score = {(x, y): float('inf') for x in self.env.x_grid[:, 0] for y in self.env.y_grid[0, :]}
        f_score[start] = self.distance(start, goal)

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                # path.append(start)
                return path[::-1]

            for neighbor in self.expand_node(current):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.distance(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None
    

    def visualize(self):
        """
        Visualize the environment and decomposed strips.
        """
        fig, ax = plt.subplots(figsize=(5, 5))

        # Plot workspace
        x, y = self.workspace.exterior.xy
        ax.plot(x, y, 'k-', label="Workspace")

        # Plot obstacles
        for obs in self.obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, 'r', alpha=0.5, label="Obstacle")

        # Plot decomposed strips
        strips = self.decompose()
        for strip in strips:
            x, y = strip.exterior.xy
            print(strip.exterior.xy)
            ax.plot(x, y, 'g--', alpha=0.7, label="Strip")

        plt.show()


# Example Usage
env = Environment(5, 5)
planner = VerticalStripDecomposition(env)
sol_path, strips, waypoints = planner.boustrophedon_path_planning()
env.visualize(sol_path, strips, waypoints)
# planner.visualize()
