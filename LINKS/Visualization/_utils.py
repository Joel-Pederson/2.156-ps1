# from ._solvers import solve_mechanism
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import xml.etree.ElementTree as etree
from svgpath2mpl import parse_path
from typing import Optional

def fetch_path():
    root = etree.parse(StringIO('<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 338"><defs><style>.cls-1{fill:#1a1a1a;stroke:#1a1a1a;stroke-linecap:round;stroke-miterlimit:10;stroke-width:20px;}</style></defs><path class="cls-1" d="M45.5,358.5l70.71-70.71M46,287.5H644m-507.61,71,70.72-70.71M223,358.5l70.71-70.71m20.18,70.72,70.71-70.71m13.67,70.7,70.71-70.71m20.19,70.72,70.71-70.71m15.84,70.71,70.71-70.71M345,39.62A121.38,121.38,0,1,1,223.62,161,121.38,121.38,0,0,1,345,39.62Z" transform="translate(-35.5 -29.62)"/></svg>')).getroot()
    view_box = root.attrib.get('viewBox')
    if view_box is not None:
        view_box = [int(x) for x in view_box.split()]
        xlim = (view_box[0], view_box[0] + view_box[2])
        ylim = (view_box[1] + view_box[3], view_box[1])
    else:
        xlim = (0, 500)
        ylim = (500, 0)
    path_elem = root.findall('.//{http://www.w3.org/2000/svg}path')[0]
    return xlim, ylim, parse_path(path_elem.attrib['d'])

def draw_mechanism(
    x0: np.ndarray,
    edges: np.ndarray,
    fixed_nodes: np.ndarray = None,
    motor: Optional[np.ndarray] = np.array([0, 1]),
    ax: Optional[plt.Axes] = None,
    highlight: Optional[int] = None,
    linkage_width: float = 2.0,
    linkage_alpha: float = 0.6,
    highlight_color: str = "#f15a24",
    joint_color: str = "#1a1a1a",
    linkage_color: str = "#1a1a1a",
    motor_color: str = "#ffc800"
):
    
    _,_,p = fetch_path()
    p.vertices -= p.vertices.mean(axis=0)
    p.vertices = (np.array([[np.cos(np.pi), -np.sin(np.pi)],[np.sin(np.pi), np.cos(np.pi)]])@p.vertices.T).T
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
        
    N = edges.max() + 1
    
    for i in range(N):
        if i in fixed_nodes:
            if i == highlight:
                ax.scatter(x0[i,0],x0[i,1],color=highlight_color,s=700,zorder=10,marker=p)
            else:
                ax.scatter(x0[i,0],x0[i,1],color=joint_color,s=700,zorder=10,marker=p)
        else:
            if i == highlight:
                ax.scatter(x0[i,0],x0[i,1],color=highlight_color,s=100,zorder=10,facecolors=highlight_color,alpha=0.7)
            else:
                ax.scatter(x0[i,0],x0[i,1],color=joint_color,s=100,zorder=10,facecolors='#ffffff',alpha=0.7)

    for e in edges:
        i,j = e
        if i == j:
            continue
        if (motor[0] == i and motor[1] == j) or (motor[0] == j and motor[1] == i):
            ax.plot([x0[i,0],x0[j,0]],[x0[i,1],x0[j,1]],color=motor_color,linewidth=linkage_width)
        else:
            ax.plot([x0[i,0],x0[j,0]],[x0[i,1],x0[j,1]],color=linkage_color,linewidth=linkage_width,alpha=linkage_alpha)

    ax.axis('equal')
    ax.axis('off')
    
    return ax