
import numpy as np
from ..Kinematics import MechanismSolver
from typing import Optional, Union
import jax
from tqdm.auto import trange

class MechanismRandomizer:
    def __init__(self, 
                 min_size: int = 6,
                 max_size: int = 20,
                 fixed_probability: float = 0.15,
                 timesteps: Optional[int] = 200,
                 device: Optional[Union[jax.Device, str]] = 'cpu'
                 ):
        '''
        A class to generate
        Parameters
        ----------
        min_size : int, optional
            Minimum number of joints in the mechanism. Default is 6.
        max_size : int, optional
            Maximum number of joints in the mechanism. Default is 20.
        fixed_probability : float, optional
            Probability of a joint being fixed. Default is 0.15.
        '''
        
        self.min_size = min_size
        self.max_size = max_size
        self.fixed_probability = fixed_probability
        
        if isinstance(device, jax.Device):
            self.device = device
        elif device=='cpu':
            self.device = jax.devices('cpu')[0]
        elif device=='gpu':
            self.device = jax.devices('gpu')[0]
        else:
            raise ValueError("Device must be 'cpu' or 'gpu' or a jax.Device instance.")
        
        self.solver = MechanismSolver(
            timesteps=timesteps, # number of timesteps to solve for --- IGNORE ---
            max_size=max_size, # maximum number of joints that will ever be simulated --- IGNORE ---
            device = self.device # device to run the simulation on (cpu or gpu)
        )
        self.solver.compile()
        
    def full_batch_compile(self,
                           target_batch_size: int = 64,
                           target_n_tests: int = 32):
        edges = np.array([[0,1],[1,3],[2,3]])

        fixed_nodes = np.array([0,2])
        motor = np.array([0,1])

        x0 = np.array([[0.3,0.2],
                       [0.3,0.3],
                       [0.6,0.2],
                       [0.6,0.4]])
        
        dummy_edges = [edges for _ in range(target_batch_size*target_n_tests)]
        dummy_fixed_nodes = [fixed_nodes for _ in range(target_batch_size*target_n_tests)]
        dummy_motor = [motor for _ in range(target_batch_size*target_n_tests)]
        dummy_x0 = [x0 for _ in range(target_batch_size*target_n_tests)]
        
        for i in trange(target_batch_size):
            self.solver(
                dummy_x0[0:(i+1)*target_n_tests],
                dummy_edges[0:(i+1)*target_n_tests],
                dummy_fixed_nodes[0:(i+1)*target_n_tests],
                dummy_motor[0:(i+1)*target_n_tests]
            )


    def batch_generate(self, batch_size: int, n:int = None, n_tests: int = 32, max_tries: int = 100):
        batched_edges = []
        batched_fixed_nodes = []
        batched_motor = []
        for _ in range(batch_size):
            edges, fixed_nodes, motor, n = self._skeleton_only(n=n)
            
            batched_edges.extend([edges for _ in range(n_tests)])
            batched_fixed_nodes.extend([fixed_nodes for _ in range(n_tests)])
            batched_motor.extend([motor for _ in range(n_tests)])
        
        
        
        solved = np.zeros((batch_size,), dtype=bool)
        mechanisms = []
        for _ in range(max_tries):
            x0s_batched = []
            for _ in range(batch_size):
                x0s = np.random.uniform(0,1,(n_tests, n, 2))
                x0s_batched.extend([x0s[i] for i in range(n_tests)])
                
            unsolved = np.where(np.logical_not(solved))[0]
            
            x0s_actual = []
            edges_actual = []
            fixed_nodes_actual = []
            motor_actual = []
            
            for i in unsolved:
                x0s_actual.extend(x0s_batched[i*n_tests:(i+1)*n_tests])
                edges_actual.extend(batched_edges[i*n_tests:(i+1)*n_tests])
                fixed_nodes_actual.extend(batched_fixed_nodes[i*n_tests:(i+1)*n_tests])
                motor_actual.extend(batched_motor[i*n_tests:(i+1)*n_tests])

            solutions = self.solver(
                x0s_actual,
                edges_actual,
                fixed_nodes_actual,
                motor_actual
            )
            
            valids = np.logical_not(np.isnan(solutions).any(axis=-1).any(axis=-1).any(axis=-1))
            
            valids = valids.reshape(-1, n_tests)
            
            solved_subsets = valids.any(axis=-1)
            
            for idx, subset_idx in enumerate(unsolved):
                if solved_subsets[idx]:
                    solved[subset_idx] = True
                    valid_idx = np.where(valids[idx])[0]
                    chosen_idx = int(np.random.choice(valid_idx, size=1)[0])
                    mechanisms.append({
                        'x0': x0s_batched[subset_idx*n_tests + chosen_idx],
                        'edges': batched_edges[subset_idx*n_tests + chosen_idx],
                        'fixed_joints': batched_fixed_nodes[subset_idx*n_tests + chosen_idx],
                        'motor': batched_motor[subset_idx*n_tests + chosen_idx]
                    })
            
            if np.all(solved):
                break
                    
        final_count = len(mechanisms)
        remaining = batch_size - final_count
        
        if remaining > 0:
            complement = self.batch_generate(remaining, n_tests, max_tries)
            mechanisms.extend(complement)
        
        return mechanisms
        
    def _skeleton_only(self,
                       n: Optional[int] = None):
        if n is None:
            n = np.random.randint(self.min_size, self.max_size+1)
        
        edges = [[0,1],[1,3],[2,3]]
    
        fixed_nodes = [0,2]
        motor = [0,1]
        
        node_types = np.random.binomial(1, self.fixed_probability, n-4)
        
        if node_types[-1] == 1:
            node_types[-1] = 0
            
        for i in range(4,n):
            if node_types[i-4] == 1:
                fixed_nodes.append(i)
            else:
                hanging_nodes = []
                for j in range(i):
                    if j not in np.array(edges)[:,0].tolist():
                        hanging_nodes.append(j)
                remaining_nodes = n - i
                
                if len(hanging_nodes) >= remaining_nodes:
                    # pick at least one from hanging nodes
                    if len(hanging_nodes) == remaining_nodes*2:
                        # must pick both from hanging nodes
                        # if all hanging nodes are fixed, we have to reset
                        if all(node in fixed_nodes for node in hanging_nodes):
                            return self._skeleton_only(n)
                        picks = np.random.choice(hanging_nodes,size=2,replace=False)
                        while picks[0] in fixed_nodes and picks[1] in fixed_nodes:
                            picks = np.random.choice(hanging_nodes,size=2,replace=False)
                    elif len(hanging_nodes) > remaining_nodes*2:
                        return self._skeleton_only(n)
                    else:
                        picks = np.array([-1,-1])
                        while not(picks[0] in hanging_nodes or picks[1] in hanging_nodes):
                            picks = np.random.choice(i,size=2,replace=False)
                            while picks[0] in fixed_nodes and picks[1] in fixed_nodes:
                                picks = np.random.choice(i,size=2,replace=False)
                else:  
                    picks = np.random.choice(i,size=2,replace=False)
                    
                    while picks[0] in fixed_nodes and picks[1] in fixed_nodes:
                        picks = np.random.choice(i,size=2,replace=False)
                
                edges.append([picks[0],i])
                edges.append([picks[1],i])
        
        edges = np.array(edges)
        fixed_nodes = np.array(fixed_nodes)
        motor = np.array(motor)
        
        return edges, fixed_nodes, motor, n
    
    def __call__(self, 
                 n: Optional[int] = None,
                 n_tests: Optional[int] = 32,
                 max_tries: Optional[int] = 100):
        
        if n is None:
            n = np.random.randint(self.min_size, self.max_size+1)
        
        edges = [[0,1],[1,3],[2,3]]
    
        fixed_nodes = [0,2]
        motor = [0,1]
        
        node_types = np.random.binomial(1, self.fixed_probability, n-4)
        
        if node_types[-1] == 1:
            node_types[-1] = 0
            
        for i in range(4,n):
            if node_types[i-4] == 1:
                fixed_nodes.append(i)
            else:
                hanging_nodes = []
                for j in range(i):
                    if j not in np.array(edges)[:,0].tolist():
                        hanging_nodes.append(j)
                remaining_nodes = n - i
                
                if len(hanging_nodes) >= remaining_nodes:
                    # pick at least one from hanging nodes
                    if len(hanging_nodes) == remaining_nodes*2:
                        # must pick both from hanging nodes
                        # if all hanging nodes are fixed, we have to reset
                        if all(node in fixed_nodes for node in hanging_nodes):
                            return self.__call__(n, n_tests, max_tries)
                        picks = np.random.choice(hanging_nodes,size=2,replace=False)
                        while picks[0] in fixed_nodes and picks[1] in fixed_nodes:
                            picks = np.random.choice(hanging_nodes,size=2,replace=False)
                    elif len(hanging_nodes) > remaining_nodes*2:
                        return self.__call__(n, n_tests, max_tries)
                    else:
                        picks = np.array([-1,-1])
                        while not(picks[0] in hanging_nodes or picks[1] in hanging_nodes):
                            picks = np.random.choice(i,size=2,replace=False)
                            while picks[0] in fixed_nodes and picks[1] in fixed_nodes:
                                picks = np.random.choice(i,size=2,replace=False)
                else:  
                    picks = np.random.choice(i,size=2,replace=False)
                    
                    while picks[0] in fixed_nodes and picks[1] in fixed_nodes:
                        picks = np.random.choice(i,size=2,replace=False)
                
                edges.append([picks[0],i])
                edges.append([picks[1],i])
        
        edges = np.array(edges)
        fixed_nodes = np.array(fixed_nodes)
        motor = np.array(motor)
        edges = [edges for _ in range(n_tests)]
        fixed_nodes = [fixed_nodes for _ in range(n_tests)]
        motor = [motor for _ in range(n_tests)]
        
        for _ in range(max_tries):
            # sample positions
            x0s = np.random.uniform(0,1,(n_tests, n, 2))
            x0s = [x0s[i] for i in range(n_tests)]
            
            solutions = self.solver(
                x0s,
                edges,
                fixed_nodes,
                motor
            )
            
            valids = np.logical_not(np.isnan(solutions).any(axis=-1).any(axis=-1).any(axis=-1))
            
            if np.any(valids):
                break
            
        if not np.any(valids):
            return self.__call__(n, n_tests, max_tries)
        
        valid_idx = np.where(valids)[0]
        chosen_idx = int(np.random.choice(valid_idx, size=1)[0])
        
        
        return {
            'x0': x0s[chosen_idx],
            'edges': edges[chosen_idx],
            'fixed_joints': fixed_nodes[chosen_idx],
            'motor': motor[chosen_idx]
        }
    
    def __getstate__(self):
        return (self.min_size,
                self.max_size,
                self.fixed_probability,
                self.solver.timesteps,
                self.device.platform
               )
        
    def __setstate__(self, state):
        self.min_size = state[0]
        self.max_size = state[1]
        self.fixed_probability = state[2]
        timesteps = state[3]
        device_str = state[4]
        
        if device_str=='cpu':
            self.device = jax.devices('cpu')[0]
        elif device_str=='gpu':
            self.device = jax.devices('gpu')[0]
        else:
            raise ValueError("Device must be 'cpu' or 'gpu'.")
        
        self.solver = MechanismSolver(
            timesteps=timesteps, # number of timesteps to solve for --- IGNORE ---
            max_size=self.max_size, # maximum number of joints that will ever be simulated --- IGNORE ---
            device = self.device # device to run the simulation on (cpu or gpu)
        )
        self.solver.compile()