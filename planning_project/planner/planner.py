import heapq
import sys, os

BASE_PATH = os.path.dirname(__file__)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from concurrent.futures import ThreadPoolExecutor

from planning_project.utils.structs import EvalMetrics
from planning_project.planner.cost_calculator import CostEstimator, CostObserver
from planning_project.planner.motion_model import motion_model
from planning_project.utils.viz import viz_terrain_props, viz_2d_map, viz_3d_map, viz_slip_models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Node:
    def __init__(self, xy_ids: tuple, node_p):
        """
        __init__:

        :param xy_ids: tuple of x- and y-axis indices
        :param node_p: parent node
        """
        self.xy_ids = xy_ids
        self.node_p = node_p

        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        """
        __eq__: check equality of two nodes' positions
        """
        return self.xy_ids == other.xy_ids

    def __lt__(self, other):
        """
        __lt__: compare values
        """
        return self.f < other.f

    def __hash__(self):
        """
        __hash__: make Node hashable for set operations
        """
        return hash(self.xy_ids)

    def __repr__(self):
        """
        __repr__: show cost for pq operation
        """
        return f'cost at node: {self.f}'


class PriorityHeapQueue:
    def __init__(self):
        """
        __init__:
        """
        self.nodes = []

    def insert(self, node):
        """
        insert: append node into open/closed list

        :param node: appended node
        """
        heapq.heappush(self.nodes, node)

    def update(self, node):
        """
        update: update cost value of the already existing node
        """
        node_id = self.nodes.index(node)
        if node.f < self.nodes[node_id].f:
            self.nodes[node_id] = node
            heapq.heapify(self.nodes)

    def pop(self):
        """
        pop: find best node and remove it from open/closed list
        """
        return heapq.heappop(self.nodes)

    def test(self, node):
        """
        test: check the node is already appended open/closed list, or not

        :param node: tested node
        """
        return node in self.nodes

class AStarPlanner:
    def __init__(self, map, smg, nn_model_dir):
        """
        __init__:

        :param map: given discretized environment map
        :param smg: slip models generator class
        :param nn_model_dir: directory of the neural network model
        """
        self.smg = smg
        self.nn_model_dir = nn_model_dir

        if map is None:
            self.map = None
            self.cost_estimator = None
            self.cost_observer = None
        else:
            self.map = map
            self.cost_estimator = None
            self.cost_observer = CostObserver(self.smg, self.map)

        self.motion = motion_model()
        self.node_start = None
        self.node_goal = None
        self.node_failed = None
        self.path = None
        self.nodes = []

        self.pred = None
        self.pred_prob = None

        # Cache for commonly used calculations
        self.heuristic_cache = {}
        self.cost_cache = {}

    def reset(self, map, start_pos: tuple, goal_pos: tuple, plan_metrics):
        """
        reset: reset position and the use of NN prediction
        """
        self.map = map
        if plan_metrics.type_model != "gtm":
            _ = self.predict(self.map.data.color.transpose(2, 0, 1).astype(np.float32))
        else:
            self.pred = None
            self.pred_prob = None

        self.cost_estimator = self.set_cost_estimator(plan_metrics)
        self.cost_observer = CostObserver(self.smg, self.map)

        self.node_start = Node(self.map.get_xy_id_from_xy_pos(start_pos[0], start_pos[1]), None)
        self.node_goal = Node(self.map.get_xy_id_from_xy_pos(goal_pos[0], goal_pos[1]), None)
        self.node_failed = None
        self.path = None
        self.nodes = []

    def predict(self, color_map):
        """
        predict: predict terrain types via trained networks
        """
        best_model = torch.load(self.nn_model_dir, map_location=torch.device(DEVICE))
        input_tensor = torch.tensor(color_map).unsqueeze(0)
        pred = best_model(input_tensor.to(DEVICE))
        self.pred_prob = pred[0].cpu().detach().numpy()
        self.pred = np.argmax(self.pred_prob, axis=0)
        return self.pred

    def set_cost_estimator(self, plan_metrics):
        """
        set_cost_estimator: set cost estimator class
        """
        if plan_metrics.type_model == "gtm":
            pred = self.map.data.t_class
        elif plan_metrics.type_model == "gsm":
            pred = self.pred
        elif plan_metrics.type_model == "gmm":
            pred = self.pred_prob
        return CostEstimator(self.smg, self.map, pred, plan_metrics)

    def search_path(self):
        """
        Optimized A* search algorithm implementation
        """
        # Use sets for faster membership testing
        open_set = {self.node_start}
        closed_set = set()
        
        # Use dictionary to store f_scores and g_scores
        f_scores = {self.node_start: self.calc_heuristic(self.node_start)}
        g_scores = {self.node_start: 0}
        
        # Use dictionary to store parent nodes
        came_from = {}
        
        while open_set:
            # Find node with lowest f_score (more efficient than using a priority queue)
            current = min(open_set, key=lambda node: f_scores.get(node, float('inf')))
            
            if current == self.node_goal:
                self.node_goal.node_p = came_from.get(current)
                break
                
            open_set.remove(current)
            closed_set.add(current)
            
            # Get neighbors using GPU acceleration
            neighbors = self.get_neighboring_nodes(current)
            
            # Process all neighbors in batch where possible
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                    
                # Calculate tentative g_score
                cost_edge, is_feasible = self.cost_estimator.calc_cost(current, neighbor)
                if not is_feasible:
                    continue
                    
                tentative_g_score = g_scores[current] + cost_edge
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_scores.get(neighbor, float('inf')):
                    continue
                
                # This path is better, record it
                came_from[neighbor] = current
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = tentative_g_score + self.calc_heuristic(neighbor)
                neighbor.g = tentative_g_score
                neighbor.f = f_scores[neighbor]
                neighbor.h = f_scores[neighbor] - tentative_g_score
                neighbor.node_p = current
        
        self.path, self.nodes = self.get_final_path()
        return self.path, self.nodes

    def is_inside_map(self, x_id: int, y_id: int):
        """
        is_inside_map: verify the given x- and y-axis indices are feasible or not
        
        :param x_id: x-axis index
        :param y_id: y-axis index
        """
        return 0 <= x_id < self.map.n and 0 <= y_id < self.map.n

    def get_neighboring_nodes(self, node):
        motions = np.array(self.motion)
        node_pos = np.array(node.xy_ids)
        x_ids = node_pos[0] + motions[:, 0]
        y_ids = node_pos[1] + motions[:, 1]
        valid_mask = (x_ids >= 0) & (x_ids < self.map.n) & (y_ids >= 0) & (y_ids < self.map.n)
        valid_x = x_ids[valid_mask]
        valid_y = y_ids[valid_mask]
        return [Node((int(x), int(y)), node) for x, y in zip(valid_x, valid_y)]

    def calc_heuristic(self, node, vel_ref: float = 0.1):
        """Cached heuristic calculation"""
        node_key = node.xy_ids
        if node_key not in self.heuristic_cache:
            pos = np.array(self.calc_pos_from_xy_id(node.xy_ids))
            pos_goal = np.array(self.calc_pos_from_xy_id(self.node_goal.xy_ids))
            self.heuristic_cache[node_key] = float(np.linalg.norm(pos - pos_goal)) / vel_ref
        return self.heuristic_cache[node_key]

    def calc_pos_from_xy_id(self, xy_ids: tuple):
        """
        calc_pos_from_xy_id: calculate positional information from indices

        :param xy_ids: x- and y-axis indices
        """
        xy_pos = np.array(xy_ids) * self.map.res + np.array([self.map.lower_left_x, self.map.lower_left_y])
        z_pos = self.map.get_value_from_xy_id(xy_ids[0], xy_ids[1])
        pos = np.append(xy_pos, z_pos)
        return pos

    def get_final_path(self):
        """
        Optimized path reconstruction
        """
        if self.node_goal.node_p is None:  # no solution found
            return None, []
            
        # Pre-allocate path array for better memory efficiency
        path = []
        nodes = []
        current = self.node_goal
        
        while current is not None:
            # Append positions and nodes in reverse order
            path.append(self.calc_pos_from_xy_id(current.xy_ids))
            nodes.append(current)
            current = current.node_p
            
        # Reverse once at the end instead of using vstack
        path = np.array(path[::-1])
        nodes.reverse()
        
        return path, nodes

    def set_final_path(self, nodes: list):
        """
        set_final_path: set final path

        :param nodes: solution of path 
        """
        if not nodes:
            path = None
        else:
            path = np.empty((0, 3), float)
            for node_c in nodes:
                path = np.vstack([path, self.calc_pos_from_xy_id(node_c.xy_ids)])
        self.path, self.nodes = path, nodes

    def execute_final_path(self):
        """
        execute_final_path: execute final path following

        """
        if self.path is None:
            metrics = EvalMetrics(is_solved=False, is_feasible=False)
            return metrics
        # init metrics
        dist = 0
        obs_time = 0 # time
        est_cost = 0 # risk-associated cost
        slips = []
        is_feasible = True
        total_traj = np.empty((0, 3), float)
        time_slips = np.empty((0, 2), float)
        for i, node_c in enumerate(self.nodes):
            if node_c == self.node_goal:
                break
            node_n = self.nodes[i+1]
            # get edge wise information
            # distance
            dist_edge, _, _ = self.cost_observer.get_edge_information(node_c, node_n)
            # estimation (planning) metrics
            est_cost_edge, _ = self.cost_estimator.calc_cost(node_c, node_n, metric="ra-time")
            # observation (execution) metrics
            obs_time_edge, is_feasible_edge, slips_edge = self.cost_observer.calc_cost(node_c, node_n)
            # path execution failed
            if not is_feasible_edge:
                self.node_failed = node_n
                obs_time = None
                is_feasible = False
                total_traj = np.vstack([total_traj, self.calc_pos_from_xy_id(node_c.xy_ids)])
                if max(slips_edge) == 1:
                    s = max(slips_edge)
                elif min(slips_edge) == -1:
                    s = min(slips_edge)
                if i != 0:
                    time_slip = np.array([[time_slips[-1, 0] + 1, s]])
                else:
                    time_slip = np.array([0, s])
                time_slips = np.vstack([time_slips, time_slip])                
            # increment total info.
            dist += dist_edge
            est_cost += est_cost_edge
            if self.node_failed is None: # only when observation is possible
                obs_time += obs_time_edge
                traj, time_slip = self.generate_traj(node_c, node_n, dist_edge / obs_time_edge)
                total_traj = np.vstack([total_traj, traj])
                if i != 0:
                    time_slip[:, 0] += time_slips[-1, 0] + 1
                time_slips = np.vstack([time_slips, time_slip])
            slips.append(max(slips_edge))
            if not is_feasible:
                break
        # add calculated information into metrics structure
        metrics = EvalMetrics(path=self.path,
                        dist=dist,
                        obs_time=obs_time,
                        est_cost=est_cost,
                        max_slip=max(slips),
                        is_solved=True,
                        is_feasible=is_feasible,
                        node_failed=self.node_failed,
                        total_traj=total_traj,
                        time_slips=time_slips)
        return metrics

    def generate_traj(self, node_c, node_n, vel):
        """
        generate_traj: generate trajectory for given edge traverse
        """
        # get positional information
        pos_c = self.calc_pos_from_xy_id(node_c.xy_ids)
        pos_n = self.calc_pos_from_xy_id(node_n.xy_ids)
        # get vector
        r_dir = pos_n - pos_c
        r_uni = r_dir / np.linalg.norm(r_dir)
        # define steps
        steps = int(np.linalg.norm(r_dir) / vel)
        vels = np.full(steps, vel)
        dr_vecs = r_uni[np.newaxis, :] * vels[:, np.newaxis]
        r_vecs = np.cumsum(dr_vecs, axis=0)
        # add first node
        traj = pos_c + r_vecs
        traj = np.vstack([pos_c, traj])
        if vel <= 0.1:
            s = (0.1 - vel) / 0.1
        else:
            s = (0.1 - vel) / vel
        slips = np.full(steps, s)
        times = np.arange(steps)
        time_slip = np.stack((times, slips), axis=1)
        return traj, time_slip

    def plot_envs(self, figsize: tuple = (18, 8), is_tf: bool = True):
        """
        plot_envs: plot 2D and 2.5 D figures and terrain classification results

        :param figsize: size of figure
        :param is_tf: existence of terrain features
        """
        sns.set()
        sns.set_style('whitegrid')
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Traversability prediction and path planning results")
        # plot 2.5 and 2d envs
        _, _ = viz_3d_map(self.map, fig=fig, rc=245, is_tf=is_tf)
        _, ax = viz_2d_map(self.map, fig=fig, rc=144)
        # plot actual and predicted models
        viz_slip_models(self.smg, fig=fig, rc_ax1=246, rc_ax2=247)
        plt.tight_layout()
        return fig, ax

    def plot_terrain_classification(self, fig):
        """
        plot_terrain_classification: show terrain classification results

        :param fig: figure
        """
        viz_terrain_props(
            vmin=0, vmax=9, n_row=2, n_col=4, fig=fig, 
            terrain_texture_map=self.map.data.color, 
            ground_truth=np.reshape(self.map.data.t_class, (self.map.n, self.map.n)), 
            prediction=np.reshape(self.pred, (self.map.n, self.map.n))
        )

    def plot_final_path(self, ax, metrics, color: str = "black", plan_type: str = "optimal planner"):
        """
        plot_final_path: plot final path

        :param ax: 
        :param metrics: metrics containing planning results
        :param color: color of path
        :param plan_type: type of planner
        """
        # visualize planned path
        if metrics.path is not None:
            if metrics.node_failed is None:
                ax.plot(metrics.path[:, 0], metrics.path[:, 1], linewidth=4, color=color, 
                        label="%s, est. cost: %.2f, obs. time: %.2f [min], max. slip: %.2f" % (plan_type, metrics.est_cost / 60, metrics.obs_time / 60, metrics.max_slip))
            else:
                ax.plot(metrics.path[:, 0], metrics.path[:, 1], linewidth=4, color=color, 
                        label='%s, failed trial' % (plan_type))
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fontsize=8)
        # visualize start/goal position
        xy_pos_start = self.calc_pos_from_xy_id(self.node_start.xy_ids)
        xy_pos_goal = self.calc_pos_from_xy_id(self.node_goal.xy_ids)
        ax.plot(xy_pos_start[0], xy_pos_start[1], marker="s", markersize=6, markerfacecolor="blue", markeredgecolor="black")
        ax.plot(xy_pos_goal[0], xy_pos_goal[1], marker="*", markersize=12, markerfacecolor="yellow", markeredgecolor="black")
        if metrics.node_failed is not None:
            xy_pos_failed = self.calc_pos_from_xy_id(metrics.node_failed.xy_ids)
            ax.plot(xy_pos_failed[0], xy_pos_failed[1], marker="X", markersize=9, markerfacecolor="red", markeredgecolor="black")
