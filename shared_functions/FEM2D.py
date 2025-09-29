import numpy as np
def FEM2D(A_h, A_v, A_d, nodes, elements, loads, constraints):
    """
    FEM2D - 2D Truss Solver with auto section assignment
    """
    # Young's modulus [kPa]
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
