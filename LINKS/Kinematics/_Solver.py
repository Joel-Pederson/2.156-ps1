from ._solvers import *
from ._kin import sort_mechanism
import numpy as np
from typing import Optional, List, Union
import jax

class MechanismSolver:
    def __init__(self,
                 timesteps: Optional[int] = 200,
                 max_size: Optional[int] = 20,
                 is_sorted: Optional[bool] = False,
                 device: Optional[Union[jax.Device, str]] = 'cpu'):
        '''
        A class to solve the kinematics of a mechanism using NumPy.
        
        Parameters
        ----------
        max_size : int, optional
            Maximum number of joints in the mechanism. Default is 20.
        is_sorted : bool, optional
            Whether the mechanisms are already sorted. Default is False.
        timesteps : int, optional
            Number of timesteps to use when solving the mechanism. Default is 200.
        device : Union[jax.Device, str], optional
            The device to use for JAX computations. Can be 'cpu', 'gpu', or a jax.Device instance. Default is 'cpu'.
        '''
        self.max_size = max_size
        self.is_sorted = is_sorted
        self.timesteps = timesteps
        
        if isinstance(device, jax.Device):
            self.device = device
        elif device=='cpu':
            self.device = jax.devices('cpu')[0]
        elif device=='gpu':
            self.device = jax.devices('gpu')[0]
        else:
            raise ValueError("Device must be 'cpu' or 'gpu' or a jax.Device instance.")
    
    def _solve(self, *args, **kwargs):
        with jax.default_device(jax.devices("cpu")[0]):
            return  dyadic_solve(*args, **kwargs)
    
    def compile(self):
        self._solve = jax.jit(self._solve, device=self.device)
    
    def __call__ (self,
                  x0: Union[np.ndarray, List[np.ndarray]],
                  edges: Union[np.ndarray, List[np.ndarray]],
                  fixed_nodes: Union[np.ndarray, List[np.ndarray]],
                  motor: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                  start_theta: Optional[float] = 0.0,
                  end_theta: Optional[float] = 2 * np.pi,
                  return_order: Optional[bool] = False):
        
        if self.is_sorted and return_order:
            raise ValueError("Cannot return order if the mechanism is already sorted.")
        
        if isinstance(x0, list):
            return self._batch_solve(x0, edges, fixed_nodes, motor, start_theta, end_theta, return_order)
        else:
            return self._solve_single(x0, edges, fixed_nodes, motor, start_theta, end_theta, return_order)
                  
    def _solve_single(self,
                      x0: np.ndarray,
                      edges: np.ndarray,
                      fixed_nodes: np.ndarray,
                      motor: Optional[np.ndarray] = None,
                      start_theta: Optional[float] = 0.0,
                      end_theta: Optional[float] = 2 * np.pi,
                      return_order: Optional[bool] = False):
        
        if motor is None:
                motor = np.array([0, 1])
                
        N = len(x0)
        if not self.is_sorted:
            edges, fixed_nodes, N, motor, x0, order, mapping = sort_mechanism(x0, edges, fixed_nodes, N, motor)

        thetas = np.linspace(start_theta, end_theta, self.timesteps)
        A = np.zeros((1, self.max_size, self.max_size), dtype=np.float64)
        A[0, edges[:, 0], edges[:, 1]] = 1.0
        A = A + np.transpose(A, (0, 2, 1))
        A = (A > 0).astype(np.float64)

        node_types = np.zeros((1, self.max_size, 1), dtype=np.float64)
        node_types[0, fixed_nodes, 0] = 1.0

        x0s = np.zeros((1, self.max_size, 2), dtype=np.float64)
        x0s[0, :x0.shape[0], :] = x0
        
        solution = self._solve(A, x0s, node_types, thetas)[0]

        if not self.is_sorted:
            solution = solution[mapping, :]
        else:
            solution = solution[:N, :]
        
        if return_order:
            return solution, order
        
        return solution
    
    def sort_batch(self,
                   x0s: List[np.ndarray],
                   edges: List[np.ndarray],
                   fixed_nodes: List[np.ndarray],
                   motors: Optional[List[np.ndarray]] = None):
        
        sorted_edges = []
        sorted_fixed_nodes = []
        sorted_x0s = []
        sorted_motors = []
        orders = []
        mappings = []
        valid = np.ones(len(x0s), dtype=bool)
        
        for i in range(len(x0s)):
            motor = motors[i] if motors is not None else np.array([0, 1])
            try:
                edges_i, fixed_nodes_i, N, motor_i, x0_i, order_i, mapping_i = sort_mechanism(x0s[i], edges[i], fixed_nodes[i], len(x0s[i]), motor)
            except:
                valid[i] = False
                edges_i = edges[i]
                fixed_nodes_i = fixed_nodes[i]
                x0_i = x0s[i]
                motor_i = motor
                order_i = np.arange(len(x0s[i]))
                mapping_i = np.arange(len(x0s[i]))
                
            sorted_edges.append(edges_i)
            sorted_fixed_nodes.append(fixed_nodes_i)
            sorted_x0s.append(x0_i)
            sorted_motors.append(motor_i)
            orders.append(order_i)
            mappings.append(mapping_i)
            
        return sorted_x0s, sorted_edges, sorted_fixed_nodes, sorted_motors, orders, mappings, valid

    def preprocess(self,
                   x0s: Union[np.ndarray, List[np.ndarray]],
                   edges: Union[np.ndarray, List[np.ndarray]],
                   fixed_nodes: Union[np.ndarray, List[np.ndarray]],
                   motors: Optional[Union[np.ndarray, List[np.ndarray]]] = None):
        
        if not isinstance(x0s, list):
            x0s = [x0s]
            edges = [edges]
            fixed_nodes = [fixed_nodes]
            motors = [motors] if motors is not None else [np.array([0, 1])]

        if not self.is_sorted:
            x0s, edges, fixed_nodes, motors, orders, mappings, valid = self.sort_batch(x0s, edges, fixed_nodes, motors)
        else:
            orders = [np.arange(len(x0s[i])) for i in range(len(x0s))]
            mappings = [np.arange(len(x0s[i])) for i in range(len(x0s))]
            valid = np.ones(len(x0s), dtype=bool)

        x0s_ = np.zeros((len(x0s), self.max_size, 2), dtype=np.float64)
        As_ = np.zeros((len(x0s), self.max_size, self.max_size), dtype=np.float64)
        node_types_ = np.zeros((len(x0s), self.max_size, 1), dtype=np.float64)
        
        for i in range(len(x0s)):
            x0s_[i, :x0s[i].shape[0], :] = x0s[i]
            As_[i, edges[i][:, 0], edges[i][:, 1]] = 1.0
            node_types_[i, fixed_nodes[i], 0] = 1.0
            node_types_[i, x0s[i].shape[0]:, 0] = 1.0

        As_ = As_ + np.transpose(As_, (0, 2, 1))
        As_ = (As_ > 0).astype(np.float64)
        
        return As_, x0s_, node_types_, motors, orders, mappings, valid
    
    def _batch_solve(self,
                    x0s: List[np.ndarray],
                    edges: List[np.ndarray],
                    fixed_nodes: List[np.ndarray],
                    motors: Optional[List[np.ndarray]] = None,
                    start_theta: Optional[float] = 0.0,
                    end_theta: Optional[float] = 2 * np.pi,
                    return_order: Optional[bool] = False):

        if not self.is_sorted:
            x0s, edges, fixed_nodes, motors, orders, mappings, valid = self.sort_batch(x0s, edges, fixed_nodes, motors)

        solutions = self._sorted_batch_solve(x0s, edges, fixed_nodes, start_theta, end_theta)

        if not self.is_sorted:
            solutions = [solutions[i][mappings[i], :] if valid[i] else solutions[i] for i in range(len(solutions))]
            
        if return_order and self.is_sorted:
            return solutions, orders

        return solutions

    def _sorted_batch_solve(self,
                    x0s: List[np.ndarray],
                    edges: List[np.ndarray],
                    fixed_nodes: List[np.ndarray],
                    start_theta: float = 0.0,
                    end_theta: float = 2 * np.pi):

        x0s_ = np.zeros((len(x0s), self.max_size, 2), dtype=np.float64)
        As_ = np.zeros((len(x0s), self.max_size, self.max_size), dtype=np.float64)
        node_types_ = np.zeros((len(x0s), self.max_size, 1), dtype=np.float64)
        thetas_ = np.linspace(start_theta, end_theta, self.timesteps)
        
        for i in range(len(x0s)):
            x0s_[i, :x0s[i].shape[0], :] = x0s[i]
            As_[i, edges[i][:, 0], edges[i][:, 1]] = 1.0
            node_types_[i, fixed_nodes[i], 0] = 1.0
            node_types_[i, x0s[i].shape[0]:, 0] = 1.0

        As_ = As_ + np.transpose(As_, (0, 2, 1))
        As_ = (As_ > 0).astype(np.float64)
        
        solutions = self._solve(As_, x0s_, node_types_, thetas_)
        
        return [solutions[i][:len(x0s[i])] for i in range(len(x0s))]