import numpy as np
import matplotlib.pyplot as plt

def FEM2D(A_h, A_v, A_d, nodes, elements, loads, constraints):
    """
    FEM2D - 2D Truss Solver with auto section assignment
    """
    # Young's modulus [kPa] [kN/m2]
    E = 210e6  
    n_nodes = nodes.shape[0]
    n_dofs = 2 * n_nodes
    n_elems = elements.shape[0]
    K_global = np.zeros((n_dofs, n_dofs))
    F_global = np.zeros(n_dofs)   # 1D force vector
    # --- Assemble stiffness matrix ---
    for e in range(n_elems):
        n1, n2 = elements[e, :] - 1
        x1, y1 = nodes[n1, :]
        x2, y2 = nodes[n2, :]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        C = (x2 - x1) / L
        S = (y2 - y1) / L
        # Detect element type
        if abs(y1 - y2) < 1e-6:
            A = A_h
        elif abs(x1 - x2) < 1e-6:
            A = A_v
        else:
            A = A_d
        # Local stiffness
        k_local = (E * A / L) * np.array([
            [ C**2,  C*S,   -C**2, -C*S ],
            [ C*S,   S**2,  -C*S,  -S**2],
            [-C**2, -C*S,   C**2,  C*S ],
            [-C*S,  -S**2,  C*S,   S**2]
        ])
        dof_map = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        for i in range(4):
            for j in range(4):
                K_global[dof_map[i], dof_map[j]] += k_local[i, j]
    # --- Apply loads ---
    for i in range(loads.shape[0]):
        dof, force = loads[i, :]
        F_global[int(dof) - 1] += force
    # --- Solve system ---
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, np.array(constraints) - 1)
    u = np.zeros(n_dofs)
    K_red = K_global[np.ix_(free_dofs, free_dofs)]
    F_red = F_global[free_dofs]
    u_red = np.linalg.solve(K_red, F_red)
    u[free_dofs] = u_red   # fill free dofs
    # --- Reactions ---
    R = K_global @ u - F_global
    reactions = R[np.array(constraints) - 1]
    # --- Axial forces ---
    axial_forces = np.zeros(n_elems)
    for e in range(n_elems):
        n1, n2 = elements[e, :] - 1
        x1, y1 = nodes[n1, :]
        x2, y2 = nodes[n2, :]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        C = (x2 - x1) / L
        S = (y2 - y1) / L
        if abs(y1 - y2) < 1e-6:
            A = A_h
        elif abs(x1 - x2) < 1e-6:
            A = A_v
        else:
            A = A_d
        dof_map = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        u_elem = u[dof_map]   # works fine (1D)
        T = np.array([-C, -S, C, S])
        axial_forces[e] = (E * A / L) * (T @ u_elem)
    return u, reactions, axial_forces

def plot_truss2(nodes, elements=None):
    """
    Plot a 2D truss structure with color-coded members and node labels.
    - Green: horizontal members
    - Yellow: vertical members
    - Blue: diagonal members
    Parameters
    ----------
    nodes : ndarray (n_nodes x 2)
        Nodal coordinates [x, y] in meters.
    elements : ndarray (n_elems x 2), optional
        Element connectivity (node1, node2). 
        If None or empty, only nodes are plotted.
        Node numbering can be 0-based or 1-based.
    """
    plt.figure(figsize=(8, 5))
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D Truss Geometry")
    # --- Plot nodes ---
    for i, (x, y) in enumerate(nodes):
        plt.plot(x, y, 'ro', markersize=8, markerfacecolor='r')
        plt.text(x + 0.1, y, f"N{i+1}", fontsize=10)
    # --- Skip element plotting if not provided ---
    if elements is None or len(elements) == 0:
        plt.show()
        return
    # Detect if elements are 1-based or 0-based
    if elements.min() == 1:
        elements = elements - 1  # convert to 0-based
    Pcount = 0  # vertical label counter
    Dcount = 0  # diagonal label counter
    # --- Plot elements ---
    for i in range(elements.shape[0]):
        n1, n2 = elements[i, :]
        x = [nodes[n1, 0], nodes[n2, 0]]
        y = [nodes[n1, 1], nodes[n2, 1]]
        xm, ym = np.mean(x), np.mean(y)
        # Determine orientation
        if abs(y[0] - y[1]) < 1e-6:  # horizontal
            color = "g"
            plt.plot(x, y, color=color, linewidth=2)
        elif abs(x[0] - x[1]) < 1e-6:  # vertical
            color = "y"
            plt.plot(x, y, color=color, linewidth=2)
            Pcount += 1
            plt.text(xm, ym, f"P{Pcount}", fontsize=10, color="k",
                     fontweight="bold", ha="center", va="center")
        else:  # diagonal
            color = "b"
            plt.plot(x, y, color=color, linewidth=2)
            Dcount += 1
            plt.text(xm, ym, f"D{Dcount}", fontsize=10, color="k",
                     fontweight="bold", ha="center", va="center")
    plt.show()

def create_roof_bracing(num_panels=4, panel_length=4.0, height=4.0):
    """
    Create geometry (nodes and elements) for a 2D X-braced truss structure.
    Each bay forms an X pattern between top and bottom chords.
    """
    # --- Nodes ---
    x_bottom = np.arange(0, (num_panels + 1) * panel_length, panel_length)
    x_top = x_bottom.copy()
    y_bottom = np.zeros_like(x_bottom)
    y_top = np.full_like(x_top, height)
    bottom_nodes = np.column_stack((x_bottom, y_bottom))
    top_nodes = np.column_stack((x_top, y_top))
    nodes = np.vstack((bottom_nodes, top_nodes))  # (N1...N(bottom), N(top)...)
    # --- Elements ---
    elements = []
    # Bottom chord (green)
    for i in range(num_panels):
        elements.append([i + 1, i + 2])  # bottom nodes
    # Top chord (green)
    for i in range(num_panels):
        elements.append([num_panels + 1 + i, num_panels + 2 + i])  # top nodes
    # Verticals (yellow)
    for i in range(num_panels + 1):
        elements.append([i + 1, num_panels + 1 + i])  # connect bottom to top
    # Diagonals (blue) — X bracing in each bay
    for i in range(num_panels):
        # D1: bottom-left → top-right
        elements.append([i + 1, num_panels + 2 + i])
        # D2: top-left → bottom-right
        elements.append([num_panels + 1 + i, i + 2])
    return np.array(nodes), np.array(elements)

def plot_truss_bracing(nodes, elements):
    """
    Plot a 2D X-braced truss (roof or wall bracing) with color-coded elements.
    Color legend:
    - Green  → horizontal members (top & bottom chords)
    - Yellow → verticals (P-members)
    - Blue   → diagonals (D-members)
    Parameters
    ----------
    nodes : ndarray (n_nodes x 2)
        Nodal coordinates [x, y].
    elements : ndarray (n_elems x 2)
        Element connectivity (1-based, as from create_roof_bracing).
    """
    plt.figure(figsize=(9, 5))
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D Roof/Wall Bracing Geometry")
    # Counters for labeling
    Pcount = 0
    Dcount = 0
    # --- Plot nodes ---
    for i in range(nodes.shape[0]):
        plt.plot(nodes[i, 0], nodes[i, 1],
                 'ro', markersize=8, markerfacecolor='r')
        plt.text(nodes[i, 0] + 0.1, nodes[i, 1],
                 f"N{i+1}", fontsize=9)
    # --- Plot elements ---
    for i in range(elements.shape[0]):
        n1, n2 = elements[i, :] - 1  # convert to 0-based
        x = [nodes[n1, 0], nodes[n2, 0]]
        y = [nodes[n1, 1], nodes[n2, 1]]
        xm, ym = np.mean(x), np.mean(y)
        dx, dy = x[1] - x[0], y[1] - y[0]
        # Determine orientation
        if abs(dy) < 1e-8:       # horizontal member
            color = "g"
            plt.plot(x, y, color=color, linewidth=2)
        elif abs(dx) < 1e-8:     # vertical post
            color = "y"
            plt.plot(x, y, color=color, linewidth=2)
            Pcount += 1
            plt.text(xm, ym, f"P{Pcount}", fontsize=9, color="k",
                     fontweight="bold", ha="center", va="center")
        else:                    # diagonal brace
            color = "b"
            plt.plot(x, y, color=color, linewidth=2)
            Dcount += 1
            plt.text(xm, ym, f"D{Dcount}", fontsize=9, color="k",
                     fontweight="bold", ha="center", va="center")
    plt.show()

def create_truss_nodes(n_panel: int, L: float, t: float):
    """
    Create nodal coordinates for a horizontal truss
    with top and bottom chords of constant height t.
    Parameters
    ----------
    n_panel : int
        Number of horizontal panels (bays)
    L : float
        Horizontal distance between verticals
    t : float
        Truss height (vertical distance between chords)
    Returns
    -------
    nodes : np.ndarray
        Array of nodal coordinates [[x0, y0], [x1, y1], ...]
    """
    # horizontal distances
    d = np.arange(0, (n_panel + 1) * L, L)
    # bottom chord (y = 0)
    x_bottom = d
    y_bottom = np.zeros_like(d)
    # top chord (y = t)
    x_top = d
    y_top = np.full_like(d, t)
    # concatenate bottom + top
    x_all = np.concatenate([x_bottom, x_top])
    y_all = np.concatenate([y_bottom, y_top])
    nodes = np.column_stack((x_all, y_all))
    return nodes

def create_X_braced_truss(n_panel: int, L: float, t: float):
    """
    Create node coordinates and connectivity matrix for a 2D X-braced truss.
    Parameters
    ----------
    n_panel : int
        Number of horizontal panels (bays)
    L : float
        Horizontal distance between verticals
    t : float
        Truss height (vertical distance between chords)
    Returns
    -------
    nodes : np.ndarray
        Array of nodal coordinates [[x, y], ...]
    elements : np.ndarray
        Element connectivity [[n1, n2], ...] (0-based indices)
    """
    # --- Node coordinates ---
    d = np.arange(0, (n_panel + 1) * L, L)
    x_bottom, y_bottom = d, np.zeros_like(d)
    x_top, y_top = d, np.full_like(d, t)
    nodes = np.column_stack((np.concatenate([x_bottom, x_top]),
                             np.concatenate([y_bottom, y_top])))
    n_bottom = n_panel + 1
    elements = []
    # --- Bottom chord ---
    for i in range(n_panel):
        elements.append([i, i + 1])
    # --- Top chord ---
    for i in range(n_panel):
        elements.append([i + n_bottom, i + n_bottom + 1])
    # --- Verticals (P-members) ---
    for i in range(n_bottom):
        elements.append([i, i + n_bottom])
    # --- Diagonals (X bracing) ---
    for i in range(n_panel):
        # Diagonal 1: bottom-left → top-right
        elements.append([i, i + 1 + n_bottom])
        # Diagonal 2: top-left → bottom-right
        elements.append([i + n_bottom, i + 1])
    return np.array(nodes), np.array(elements)

def create_vertical_bracing_nodes(nf: int, h: float, L: float):
    """
    Create nodal coordinates for a vertical wall bracing truss.
    Parameters
    ----------
    nf : int
        Number of floors (vertical panels)
    h : float
        Height of each floor (m)
    L : float
        Horizontal distance between left and right chords (m)
    Returns
    -------
    nodes : np.ndarray
        Array of nodal coordinates [[x, y], ...] (0-based indexing)
    """
    # Vertical distances (heights)
    y_levels = np.arange(0, (nf + 1) * h, h)
    # Left chord nodes (x=0)
    x_left = np.zeros_like(y_levels)
    y_left = y_levels
    # Right chord nodes (x=L)
    x_right = np.full_like(y_levels, L)
    y_right = y_levels
    # Combine into full node set
    nodes = np.column_stack((
        np.concatenate([x_left, x_right]),
        np.concatenate([y_left, y_right])
    ))
    return nodes

def create_vertical_bracing_elements(nf: int):
    """
    Create element connectivity for a vertical X-braced wall truss.
    Parameters
    ----------
    nf : int
        Number of floors (vertical panels)
    Returns
    -------
    elements : np.ndarray
        Array of element connectivity [[n1, n2], ...] (0-based indexing)
    """
    elements = []
    n_left = nf + 1  # number of nodes on one side
    # --- Left vertical chord ---
    for i in range(nf):
        elements.append([i, i + 1])
    # --- Right vertical chord ---
    for i in range(nf):
        elements.append([i + n_left, i + n_left + 1])
    # --- Horizontal ties (no ground level) ---
    for i in range(1, n_left):  # start from level 1, skip ground (i=0)
        elements.append([i, i + n_left])
    # --- Diagonals for each floor bay ---
    for i in range(nf):
        # D1: left lower → right upper
        elements.append([i, i + n_left + 1])
        # D2: right lower → left upper
        elements.append([i + n_left, i + 1])
    return np.array(elements)

def create_vertical_bracing_truss(nf: int, h: float, L: float):
    """
    Full geometry generation for a vertical wall X-braced truss.
    Returns both node coordinates and element connectivity.
    """
    nodes = create_vertical_bracing_nodes(nf, h, L)
    elements = create_vertical_bracing_elements(nf)
    return nodes, elements

