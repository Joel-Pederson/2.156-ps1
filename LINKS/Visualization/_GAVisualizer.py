from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Union, List
from . import MechanismVisualizer
from ..Geometry import CurveEngine
from ..Kinematics import MechanismSolver

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    Parameters
    ----------
    costs:      numpy array [N,n_objectives]
                Objective-wise performance values
    return_mask:bool
                If True, the function returns a mask of dominated points, else it returns the indices of efficient points.
    Returns
    -------
    is_efficient:   numpy array [N]
                    If return_mask is True, this is an array boolean values indicating whether each point in the input `costs` is Pareto efficient.
                    If return_mask is False, this is an array of indices of the points in the input `costs` array that are Pareto efficient.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

class GAVisualizer:
    def __init__(self,
                 timesteps: Optional[int] = 200,
                 max_size: Optional[int] = 20,
                 scaled: Optional[bool] = False):
        
        self.visualizer = MechanismVisualizer(
            timesteps=timesteps,
            max_size=max_size
        )


        self.solver = MechanismSolver(timesteps=timesteps,
                                      max_size=max_size,
                                      device='cpu')

        self.CurveEngine = CurveEngine(resolution=timesteps,
                                       normalize_scale=scaled,
                                       device='cpu')
    
    def plot_HV(self,
                F: Union[np.ndarray, list],
                ref: Union[np.ndarray, list],
                objective_labels: Optional[list] = None,
                point_color: Optional[str] = 'navy',
                ref_color: Optional[str] = 'maroon',
                fill_color: Optional[str] = "#ff9900",
                fill_alpha: Optional[float] = 0.5,
                ax = None):

        F = np.array(F)
        ref = np.array(ref)

        valids = np.logical_and(F[:,0] < ref[0], F[:,1] < ref[1])
        F = F[valids]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))

        # Plot the designs
        ax.scatter(F[:,1],F[:,0], color=point_color)

        # plot the reference point
        ax.scatter(ref[1],ref[0], color=ref_color)

        # plot labels
        if objective_labels is not None:
            ax.set_ylabel(objective_labels[0])
            ax.set_xlabel(objective_labels[1])
        else:
            ax.set_xlabel('Objective 2')
            ax.set_ylabel('Objective 1')

        
        pareto_efficients = is_pareto_efficient(F)
        F_p = F[pareto_efficients]
        
        # sort designs and append reference point
        sorted_performance = F_p[np.argsort(F_p[:,1])]
        sorted_performance = np.concatenate([sorted_performance,[ref]])

        #create "ghost points" for inner corners
        inner_corners = np.stack([sorted_performance[:,0], np.roll(sorted_performance[:,1], -1)]).T

        #Interleave designs and ghost points
        final = np.empty((sorted_performance.shape[0]*2, 2))
        final[::2,:] = sorted_performance
        final[1::2,:] = inner_corners

        #Create filled polygon
        ax.fill(final[:,1],final[:,0],color=fill_color, alpha=fill_alpha)
        
        return ax

    def plot_pareto_efficient(self,
                              F: Union[np.ndarray, list],
                              population: List[dict],
                              target_curve : np.ndarray,
                              objective_labels: Optional[list] = None):
        
        ind = is_pareto_efficient(F)
        F_p = F[ind]
        
        ind_sorter = np.argsort(F_p[:,0])
        F_p = F_p[ind_sorter]
        
        ind = np.where(ind)[0]
        
        fig, axs = plt.subplots(F_p.shape[0], 3,figsize=(15,5*F_p.shape[0]))
        
        if objective_labels is not None:
            label1 = objective_labels[0]
            label2 = objective_labels[1]
        else:
            label1 = 'Objective 1'
            label2 = 'Objective 2'
        
        for i in range(F_p.shape[0]):
            idx = ind[ind_sorter][i]
            self.visualizer(
                population[idx]['x0'],
                population[idx]['edges'],
                population[idx]['fixed_joints'],
                population[idx]['motor'],
                ax=axs[i,0],
                highlight=population[idx].get('target_joint', None)
            )
            
            solution, order = self.solver(
                population[idx]['x0'],
                population[idx]['edges'],
                population[idx]['fixed_joints'],
                population[idx]['motor'],
                return_order=True
            )
            
            if population[idx].get('target_joint', None) is None:
                target = order[-1]
            else:
                target = population[idx]['target_joint']
            
            traced_curve = solution[target]
            
            self.CurveEngine.visualize_single_comparison(
                traced_curve,
                target_curve,
                ax=axs[i,1]
            )
            axs[i,1].set_title(f"{label1}: {F_p[i,0]:.5f} | {label2}: {F_p[i,1]:.2f}")

            axs[i,2].scatter(F_p[:,1],F_p[:,0], color='royalblue')
            axs[i,2].set_xlabel(label2)
            axs[i,2].set_ylabel(label1)
            axs[i,2].scatter([F_p[i,1]],[F_p[i,0]],color="tomato")
            #hide top and right axis
            axs[i,2].spines['top'].set_visible(False)
            axs[i,2].spines['right'].set_visible(False)
        
        