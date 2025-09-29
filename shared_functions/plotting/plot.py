import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
def plot(nodes, elements, u=None, axial_forces=None, constraints=None, scale=1.0):
    """
    Plot truss structure with optional deformed shape, element forces, 
    node coloring, and constraint arrows.
    Parameters
    ----------
    nodes : ndarray (N, 2)
        Node coordinates [x, y].
    elements : ndarray (M, 2), 1-based node indices (MATLAB style).
    u : ndarray (2N,), optional
        Global displacement vector.
    axial_forces : ndarray (M,), optional
        Axial force per element (positive = tension, negative = compression).
    constraints : list of int, optional
        List of constrained DOFs (1-based, like MATLAB).
    scale : float
        Scale factor for deformation (only used if u is given).
    """
    # pick which coordinates to plot
    if u is None:
        coords = nodes
        title = "Original Truss"
        default_color = "k-"
    else:
        coords = nodes + scale * u.reshape(-1, 2)
        title = f"Deformed Truss (scale={scale})"
        default_color = None
    plt.figure()
    # --- Draw elements ---
    for i, e in enumerate(elements):
        n1, n2 = e[0] - 1, e[1] - 1  # shift to 0-based
        x = [coords[n1, 0], coords[n2, 0]]
        y = [coords[n1, 1], coords[n2, 1]]
        if u is None:
            plt.plot(x, y, default_color, linewidth=1.5)
        else:
            if axial_forces is None:
                plt.plot(x, y, "k-", linewidth=1.5)
            else:
                force_val = axial_forces[i]
                width = 1.5 + 4.0 * (abs(force_val) / np.max(np.abs(axial_forces)))
                color = "b-" if force_val >= 0 else "r-"
                plt.plot(x, y, color, linewidth=width)
    # --- Node colors based on constraints (always applied) ---
    N = nodes.shape[0]
    if constraints is None:
        constraints = []
    node_colors = []
    for i in range(N):
        dof_x = 2*i + 1  # 1-based
        dof_y = 2*i + 2  # 1-based
        fixed_x = dof_x in constraints
        fixed_y = dof_y in constraints
        if not fixed_x and not fixed_y:
            node_colors.append("green")   # free
        elif fixed_x and not fixed_y:
            node_colors.append("yellow")  # x fixed
        elif not fixed_x and fixed_y:
            node_colors.append("brown")   # y fixed
        elif fixed_x and fixed_y:
            node_colors.append("purple")  # fully fixed
    # Plot nodes with colors
    for i, (x, y) in enumerate(coords):
        plt.plot(x, y, "o", markersize=8, color=node_colors[i])
        plt.text(x, y, f"{i+1}", fontsize=9, ha="center", va="bottom", color="black")
    # --- Draw constraint arrows ---
    arrow_size = 0.3
    for i, (x, y) in enumerate(coords):
        dof_x = 2*i + 1
        dof_y = 2*i + 2
        if dof_x in constraints:
            plt.arrow(x, y, -arrow_size, 0, head_width=0.15, head_length=0.15,
                      fc="k", ec="k")
        if dof_y in constraints:
            plt.arrow(x, y, 0, -arrow_size, head_width=0.15, head_length=0.15,
                      fc="k", ec="k")
    # --- Legends ---
    handles = []
    if u is not None and axial_forces is not None:
        tension_line = mlines.Line2D([], [], color='blue', label='Tension')
        compression_line = mlines.Line2D([], [], color='red', label='Compression')
        handles.extend([tension_line, compression_line])
    # Node legend (always shown)
    node_legend = [
        mlines.Line2D([], [], color='green', marker='o', linestyle='None', label='Free Node'),
        mlines.Line2D([], [], color='yellow', marker='o', linestyle='None', label='X Fixed'),
        mlines.Line2D([], [], color='brown', marker='o', linestyle='None', label='Y Fixed'),
        mlines.Line2D([], [], color='purple', marker='o', linestyle='None', label='XY Fixed')
    ]
    handles.extend(node_legend)
    plt.legend(handles=handles, loc="best")
    plt.title(title)
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()










    