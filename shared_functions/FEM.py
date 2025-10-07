import numpy as np
import matplotlib.pyplot as plt

def FEM2D(A_h, A_v, A_d, nodes, elements, loads, constraints, E=210e6):
    """
    FEM2D - Simplified 2D Truss Solver (Pythonic convention) [kN, m units]
    Parameters
    ----------
    A_h, A_v, A_d : float
        Cross-sectional areas (m²) for horizontal, vertical, and diagonal members.
    nodes : ndarray (n_nodes x 2)
        Node coordinates [x, y] in meters.
    elements : ndarray (n_elems x 2)
        Element connectivity (0-based node indices).
    loads : ndarray (n_loads x 2)
        External loads: [DOF_index, force_value] in kN.
    constraints : list or ndarray
        Constrained DOFs (0-based global DOF indices).
    E : float
        Young’s modulus (kN/m²). Default = 210000 kN/m² ≈ 210 GPa.
    Returns
    -------
    u : ndarray (n_dofs,)
        Global displacement vector (m).
    reactions : ndarray
        Reaction forces at constrained DOFs (kN).
    axial_forces : ndarray (n_elems,)
        Axial force in each truss element (kN).
    elem_types : list of str
        List of element types: 'H', 'V', or 'D'.
    """
    n_nodes = nodes.shape[0]
    n_elems = elements.shape[0]
    n_dofs = 2 * n_nodes
    # --- Global matrices ---
    K_global = np.zeros((n_dofs, n_dofs))
    F_global = np.zeros(n_dofs)
    elem_types = []  # Track element type: 'H', 'V', 'D'
    # --- Assemble stiffness matrix ---
    for e in range(n_elems):
        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        C = (x2 - x1) / L
        S = (y2 - y1) / L
        # Assign correct area and type
        if abs(y1 - y2) < 1e-9:
            A = A_h
            elem_types.append('H')
        elif abs(x1 - x2) < 1e-9:
            A = A_v
            elem_types.append('V')
        else:
            A = A_d
            elem_types.append('D')
        # Local stiffness matrix
        k_local = (E * A / L) * np.array([
            [ C*C,  C*S, -C*C, -C*S],
            [ C*S,  S*S, -C*S, -S*S],
            [-C*C, -C*S,  C*C,  C*S],
            [-C*S, -S*S,  C*S,  S*S]
        ])
        # DOF mapping
        dof = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        K_global[np.ix_(dof, dof)] += k_local
    # --- Apply loads (in kN) ---
    for dof, value in loads:
        F_global[int(dof)] += value
    # --- Solve system ---
    all_dofs = np.arange(n_dofs)
    free_dofs = np.setdiff1d(all_dofs, constraints)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_ff = F_global[free_dofs]
    u = np.zeros(n_dofs)
    u[free_dofs] = np.linalg.solve(K_ff, F_ff)
    # --- Reactions ---
    R = K_global @ u - F_global
    reactions = R[constraints]
    # --- Axial forces ---
    axial_forces = np.zeros(n_elems)
    for e in range(n_elems):
        n1, n2 = elements[e]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        C = (x2 - x1) / L
        S = (y2 - y1) / L
        # Reuse same A logic
        if elem_types[e] == 'H':
            A = A_h
        elif elem_types[e] == 'V':
            A = A_v
        else:
            A = A_d
        dof = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        u_elem = u[dof]
        T = np.array([-C, -S, C, S])
        axial_forces[e] = (E * A / L) * (T @ u_elem)
    return u, reactions, axial_forces, elem_types

def plot_truss(nodes, elements=None, loads=None, constraints=None,load_scale=0.2):
    """
    Plot a 2D truss structure with color-coded members, boundary conditions, and nodal loads.
    Colors:
        - Green: horizontal members
        - Yellow: vertical members
        - Blue: diagonal members
        - Purple: double fixed node (X and Y)
        - Green: fixed X
        - Brown: fixed Y
        - Black: free node
        - Red arrows: applied loads
    Parameters
    ----------
    nodes : ndarray (n_nodes x 2)
        Nodal coordinates [x, y].
    elements : ndarray (n_elems x 2), optional
        Element connectivity (node1, node2). 0- or 1-based indexing.
    constraints : list or ndarray, optional
        List of fixed DOFs (0-based). Example [0, 1, 7] means:
            - Node 0 fixed in X and Y
            - Node 3 fixed in Y (since 7 = 3*2 + 1)
    loads : ndarray or list, optional
        [[DOF_index, force_value], ...] — global DOF loads (FEM2D2 style)
    load_scale : float
        Arrow scaling for load visualization.
    """
    plt.figure(figsize=(8, 5))
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("2D Truss Geometry")
    n_nodes = len(nodes)
    n_dofs = n_nodes * 2
    # --- Build constraint map per node ---
    node_constraints = np.zeros((n_nodes, 2), dtype=bool)  # [Fx_fixed, Fy_fixed]
    if constraints is not None:
        for dof in constraints:
            node = dof // 2
            dirn = dof % 2  # 0 = X, 1 = Y
            if node < n_nodes:
                node_constraints[node, dirn] = True
    # --- Node color logic ---
    for i, (x, y) in enumerate(nodes):
        fixed_x, fixed_y = node_constraints[i]
        if fixed_x and fixed_y:
            color = 'purple'  # double fixed
        elif fixed_x:
            color = 'green'   # fixed X
        elif fixed_y:
            color = 'brown'   # fixed Y
        else:
            color = 'black'   # free
        plt.plot(x, y, 'o', color=color, markersize=8)
        plt.text(x + 0.15, y, f"N{i+1}", fontsize=10, color='k')
    # --- Adjust element indexing if 1-based ---
    if elements is not None and len(elements) > 0:
        if elements.min() == 1:
            elements = elements - 1
        # --- Plot members and label verticals/diagonals ---
        vert_count = 0
        diag_count = 0
        for i in range(elements.shape[0]):
            n1, n2 = elements[i, :]
            x = [nodes[n1, 0], nodes[n2, 0]]
            y = [nodes[n1, 1], nodes[n2, 1]]
            xm, ym = np.mean(x), np.mean(y)
            # Determine orientation
            if abs(y[0] - y[1]) < 1e-6:
                color = "g"  # horizontal
                label = None
            elif abs(x[0] - x[1]) < 1e-6:
                color = "y"  # vertical
                vert_count += 1
                label = f"P{vert_count}"
            else:
                color = "b"  # diagonal
                diag_count += 1
                label = f"D{diag_count}"
            plt.plot(x, y, color=color, linewidth=2)
            if label:
                plt.text(xm, ym, label, fontsize=10, color="k",
                         fontweight="bold", ha="center", va="center")
    # --- Plot loads (DOF-based only) ---
    if loads is not None and len(loads) > 0:
        node_loads = np.zeros((n_nodes, 2))
        for dof, val in loads:
            dof = int(dof)
            node = dof // 2
            dirn = dof % 2
            if node < n_nodes:
                node_loads[node, dirn] += val
        for i, (Fx, Fy) in enumerate(node_loads):
            if abs(Fx) > 1e-9 or abs(Fy) > 1e-9:
                x, y = nodes[i]
                plt.arrow(
                    x, y,
                    load_scale * Fx, load_scale * Fy,
                    head_width=0.15, head_length=0.25,
                    fc='red', ec='red', linewidth=1.5, zorder=5
                )
    # --- Legend ---
    legend_items = [
        plt.Line2D([], [], color='g', lw=2, label='Horizontal member'),
        plt.Line2D([], [], color='y', lw=2, label='Vertical member (Pn)'),
        plt.Line2D([], [], color='b', lw=2, label='Diagonal member (Dn)'),
        plt.Line2D([], [], color='purple', marker='o', linestyle='None', label='Double fixed'),
        plt.Line2D([], [], color='green', marker='o', linestyle='None', label='Fixed X'),
        plt.Line2D([], [], color='brown', marker='o', linestyle='None', label='Fixed Y'),
        plt.Line2D([], [], color='black', marker='o', linestyle='None', label='Free'),
        plt.Line2D([], [], color='red', marker=r'$\rightarrow$', linestyle='None', label='Load')
    ]
    plt.legend(handles=legend_items, loc='upper right', fontsize=8)
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

def create_X_horizontal_truss(n_panel: int, L: float, t: float):
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


