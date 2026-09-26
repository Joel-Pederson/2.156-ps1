import numpy as np
from typing import Optional, Union, List

def dyadic_path(
    edges : np.ndarray,
    fixed_nodes : np.ndarray,
    N : int,
    motor : Optional[Union[np.ndarray, List[int]]] = [0, 1]
):
    knowns = np.append(fixed_nodes, motor[1])
    unkowns = np.arange(N)
    unkowns = unkowns[~np.isin(unkowns, knowns)]
    
    full_edges = np.concatenate((edges, np.fliplr(edges)))
    full_edges = np.unique(full_edges, axis=0)
    full_edges = full_edges[np.argsort(full_edges[:,0])]
    
    ptr = np.searchsorted(full_edges[:,0], np.arange(N))
    ptr = np.append(ptr, full_edges.shape[0])
    
    counter = 0
    path = np.zeros((unkowns.shape[0], 3), dtype=edges.dtype)
    pc = 0
    while unkowns.shape[0] != 0:
        if counter == unkowns.shape[0]:
            # Non dyadic or DOF larger than 1
            return path, 1
        
        n = unkowns[counter]
        ne = full_edges[ptr[n]:ptr[n+1], 1]

        kne = ne[np.isin(ne,knowns)]
        
        if kne.shape[0] == 2:
            path[pc, 0] = n
            path[pc, 1] = kne[0]
            path[pc, 2] = kne[1]
            pc += 1
            knowns = np.insert(knowns, 0, n)
            unkowns = np.delete(unkowns, counter)
            counter = 0
        elif kne.shape[0] > 2:
            #redundant or overconstraint
            return path, 2
        else:
            counter += 1

    return path, 0

def sort_mechanism(
    x0 : np.ndarray,
    edges : np.ndarray,
    fixed_nodes : np.ndarray,
    N : int,
    motor : Optional[Union[np.ndarray, List[int]]] = [0, 1]
):
    path, status = dyadic_path(edges, fixed_nodes, N, motor)
    
    if status != 0:
        if status == 1:
            raise ValueError("Mechanism is not dyadic or has DOF larger than 1")
        elif status == 2:
            raise ValueError(f"Mechanism has redundant linkages or is overconstrained")

    order = np.append(
        np.insert(
            fixed_nodes[fixed_nodes != motor[0]],
            0,
            motor
        ),
        path[:, 0]
    )
    
    mapping = np.argsort(order)
    
    fixed_nodes_sorted = mapping[fixed_nodes]
    edges_sorted = np.zeros_like(edges)
    edges_sorted[:, 0] = mapping[edges[:, 0]]
    edges_sorted[:, 1] = mapping[edges[:, 1]]
    x0_sorted = x0[order].squeeze()
    

    return edges_sorted, fixed_nodes_sorted, N, np.array([0,1]), x0_sorted, order, mapping