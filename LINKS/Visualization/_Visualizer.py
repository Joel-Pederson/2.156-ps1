from matplotlib import pyplot as plt
import numpy as np
from ..Kinematics import MechanismSolver
from typing import Optional
from ._utils import draw_mechanism

class MechanismVisualizer:
    def __init__(self, 
                 solve = True,
                 timesteps = 200,
                 max_size = 20,
                 joint_color = "#1a1a1a",
                 linkage_color = "#1a1a1a",
                 linkage_width = 4.5,
                 linkage_alpha = 0.7,
                 motor_color = "#ffc800",
                 highlight_color = "#f15a24",
                 curve_color = "#0078a7",
                 curve_alpha = 1.0,
                 curve_width = 1.5,
                 curve_pattern = '--',
                 highlight_curve_pattern = '-',
                 highlight = 'auto'):
        '''
        A class to visualize mechanisms and their motion.
        Parameters
        ----------
        solve : bool, optional
            Whether to solve the mechanism motion before visualization. Default is True.
        timesteps : int, optional
            Number of timesteps to use when solving the mechanism. Default is 200.
        max_size : int, optional
            Maximum number of joints in the mechanism. Default is 20.
        joint_color : str, optional
            Color of the joints. Default is "#1a1a1a".
        linkage_color : str, optional
            Color of the linkages. Default is "#1a1a1a".
        linkage_width : float, optional
            Width of the linkages. Default is 4.5.
        linkage_alpha : float, optional
            Alpha (transparency) of the linkages. Default is 0.7.
        motor_color : str, optional
            Color of the motor linkage. Default is "#ffc800".
        highlight_color : str, optional
            Color to use for highlighting a joint or curve. Default is "#f15a24".
        curve_color : str, optional
            Color of the trajectory curves. Default is "#0078a7".
        curve_alpha : float, optional
            Alpha (transparency) of the trajectory curves. Default is 1.0.
        curve_width : float, optional
            Width of the trajectory curves. Default is 1.5.
        curve_pattern : str, optional
            Line style for the trajectory curves. Default is '--'.
        highlight_curve_pattern : str, optional
            Line style for the highlighted trajectory curve. Default is '-'.
        highlight : Union[str, int, None], optional
            Which joint to highlight. If 'auto', highlights the terminal joint if solving is enabled.
        '''
        
        self.solver = MechanismSolver(max_size=max_size, timesteps=timesteps, device='cpu')
        self.solve = solve
        self.timesteps = timesteps
        self.joint_color = joint_color
        self.linkage_color = linkage_color
        self.motor_color = motor_color
        self.highlight_color = highlight_color
        self.curve_color = curve_color
        self.curve_pattern = curve_pattern
        self.highlight_curve_pattern = highlight_curve_pattern
        self.curve_alpha = curve_alpha
        self.linkage_width = linkage_width
        self.linkage_alpha = linkage_alpha
        self.curve_width = curve_width
        
        if highlight == 'auto' and solve:
            self.highlight = 'terminal'
        elif highlight == 'auto' and not solve:
            self.highlight = None
        else:
            self.highlight = highlight
            
    def __call__(self,
                 x0: np.ndarray,
                 edges: np.ndarray,
                 fixed_joints: np.ndarray = None,
                 motor: Optional[np.ndarray] = np.array([0, 1]),
                 ax: Optional[plt.Axes] = None,
                 highlight: Optional[int] = None,
                 solution=None):
        '''
        Visualize the mechanism and its motion (if solve is True or solution is provided).
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial positions of the joints, shape (N, 2).
        edges : np.ndarray
            Edges defining the linkages, shape (M, 2).
        fixed_nodes : np.ndarray, optional
            Indices of fixed joints, shape (K,).
        motor : np.ndarray, optional
            Indices defining the motor linkage, shape (2,). Default is np.array([0, 1]).
        ax : plt.Axes, optional
            Matplotlib Axes to draw on. If None, a new figure and axes are created
        solution : np.ndarray, optional
            Precomputed solution of joint positions over time, shape (T, N, 2). If provided, this is used instead of solving.
        '''
        
        highlight = self.highlight if highlight is None else highlight
        if self.solve and solution is None:
            solution, order = self.solver(x0, edges, fixed_joints, motor, return_order=True)
            if self.highlight == 'terminal' and not isinstance(highlight, int):
                highlight = order[-1]
        if solution is not None and highlight == 'terminal':
            highlight = solution.shape[0]-1 # assume last is terminal position
        
        ax = draw_mechanism(
            x0,
            edges,
            fixed_nodes=fixed_joints,
            motor=motor,
            ax=ax,
            highlight=highlight,
            highlight_color=self.highlight_color,
            joint_color=self.joint_color,
            linkage_color=self.linkage_color,
            motor_color=self.motor_color,
            linkage_alpha=self.linkage_alpha,
            linkage_width=self.linkage_width
        )
        
        if self.solve or solution is not None:
            for i in range(solution.shape[0]):
                if i == highlight:
                    ax.plot(solution[i, :, 0], solution[i, :, 1], color=self.highlight_color, linestyle=self.highlight_curve_pattern, linewidth=self.curve_width*2, alpha=self.curve_alpha)
                else:
                    ax.plot(solution[i, :, 0], solution[i, :, 1], color=self.curve_color, linestyle=self.curve_pattern, linewidth=self.curve_width, alpha=self.curve_alpha)

        return ax