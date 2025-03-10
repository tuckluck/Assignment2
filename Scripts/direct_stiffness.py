
import math as mt
import numpy as np
import scipy as sc

            

def local_elastic_stiffness_matrix_3D_beam(E: float, nu: float, A: float, L: float, Iy: float, Iz: float, J: float) -> np.ndarray:
    """
    local element elastic stiffness matrix
    source: p. 73 of McGuire's Matrix Structural Analysis 2nd Edition
    Given:
        material and geometric parameters:
            A, L, Iy, Iz, J, nu, E
    Context:
        load vector:
            [Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2]
        DOF vector:
            [u1, v1, w1, th_x1, th_y1, th_z1, u2, v2, w2, th_x2, th_y2, th_z2]
        Equation:
            [load vector] = [stiffness matrix] @ [DOF vector]
    Returns:
        12 x 12 elastic stiffness matrix k_e
    """
    k_e = np.zeros((12, 12))
    # Axial terms - extension of local x axis
    axial_stiffness = E * A / L
    k_e[0, 0] = axial_stiffness
    k_e[0, 6] = -axial_stiffness
    k_e[6, 0] = -axial_stiffness
    k_e[6, 6] = axial_stiffness
    # Torsion terms - rotation about local x axis
    torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
    k_e[3, 3] = torsional_stiffness
    k_e[3, 9] = -torsional_stiffness
    k_e[9, 3] = -torsional_stiffness
    k_e[9, 9] = torsional_stiffness
    # Bending terms - bending about local z axis
    k_e[1, 1] = E * 12.0 * Iz / L ** 3.0
    k_e[1, 7] = E * -12.0 * Iz / L ** 3.0
    k_e[7, 1] = E * -12.0 * Iz / L ** 3.0
    k_e[7, 7] = E * 12.0 * Iz / L ** 3.0
    k_e[1, 5] = E * 6.0 * Iz / L ** 2.0
    k_e[5, 1] = E * 6.0 * Iz / L ** 2.0
    k_e[1, 11] = E * 6.0 * Iz / L ** 2.0
    k_e[11, 1] = E * 6.0 * Iz / L ** 2.0
    k_e[5, 7] = E * -6.0 * Iz / L ** 2.0
    k_e[7, 5] = E * -6.0 * Iz / L ** 2.0
    k_e[7, 11] = E * -6.0 * Iz / L ** 2.0
    k_e[11, 7] = E * -6.0 * Iz / L ** 2.0
    k_e[5, 5] = E * 4.0 * Iz / L
    k_e[11, 11] = E * 4.0 * Iz / L
    k_e[5, 11] = E * 2.0 * Iz / L
    k_e[11, 5] = E * 2.0 * Iz / L
    # Bending terms - bending about local y axis
    k_e[2, 2] = E * 12.0 * Iy / L ** 3.0
    k_e[2, 8] = E * -12.0 * Iy / L ** 3.0
    k_e[8, 2] = E * -12.0 * Iy / L ** 3.0
    k_e[8, 8] = E * 12.0 * Iy / L ** 3.0
    k_e[2, 4] = E * -6.0 * Iy / L ** 2.0
    k_e[4, 2] = E * -6.0 * Iy / L ** 2.0
    k_e[2, 10] = E * -6.0 * Iy / L ** 2.0
    k_e[10, 2] = E * -6.0 * Iy / L ** 2.0
    k_e[4, 8] = E * 6.0 * Iy / L ** 2.0
    k_e[8, 4] = E * 6.0 * Iy / L ** 2.0
    k_e[8, 10] = E * 6.0 * Iy / L ** 2.0
    k_e[10, 8] = E * 6.0 * Iy / L ** 2.0
    k_e[4, 4] = E * 4.0 * Iy / L
    k_e[10, 10] = E * 4.0 * Iy / L
    k_e[4, 10] = E * 2.0 * Iy / L
    k_e[10, 4] = E * 2.0 * Iy / L
    return k_e


def check_unit_vector(vec: np.ndarray):
    """
    """
    if np.isclose(np.linalg.norm(vec), 1.0):
        return
    else:
        raise ValueError("Expected a unit vector for reference vector.")


def check_parallel(vec_1: np.ndarray, vec_2: np.ndarray):
    """
    """
    if np.isclose(np.linalg.norm(np.cross(vec_1, vec_2)), 0.0):
        raise ValueError("Reference vector is parallel to beam axis.")
    else:
        return


def rotation_matrix_3D(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, v_temp: np.ndarray = None):
    """
    3D rotation matrix
    source: Chapter 5.1 of McGuire's Matrix Structural Analysis 2nd Edition
    Given:
        x, y, z coordinates of the ends of two beams: x1, y1, z1, x2, y2, z2
        optional: reference z vector direction v_temp to orthonormalize the local y and z axis
            if v_temp is not given, VVVV
    Compute:
        where l, m, n are defined as direction cosines:
        gamma = [[lx'=cos alpha_x', mx'=cos beta_x', nx'=cos gamma_x'],
                 [ly'=cos alpha_y', my'=cos beta_y', ny'=cos gamma_y'],
                 [lz'=cos alpha_z', mz'=cos beta_z', nz'=cos gamma_z']]
    """
    L = np.sqrt((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0 + (z2 - z1) ** 2.0)
    lxp = (x2 - x1) / L
    mxp = (y2 - y1) / L
    nxp = (z2 - z1) / L
    local_x = np.asarray([lxp, mxp, nxp])

    # choose a vector to orthonormalize the y axis if one is not given
    if v_temp is None:
        # if the beam is oriented vertically, switch to the global y axis
        if np.isclose(lxp, 0.0) and np.isclose(mxp, 0.0):
            v_temp = np.array([0, 1.0, 0.0])
        else:
            # otherwise use the global z axis
            v_temp = np.array([0, 0, 1.0])
    else:
        # check to make sure that given v_temp is a unit vector
        check_unit_vector(v_temp)
        # check to make sure that given v_temp is not parallel to the local x axis
        check_parallel(local_x, v_temp)
    
    # compute the local y axis
    local_y = np.cross(v_temp, local_x)
    local_y = local_y / np.linalg.norm(local_y)

    # compute the local z axis
    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)

    # assemble R
    gamma = np.vstack((local_x, local_y, local_z))
    
    return gamma


def transformation_matrix_3D(gamma: np.ndarray) -> np.ndarray:
    """
    3D transformation matrix
    source: Chapter 5.1 of McGuire's Matrix Structural Analysis 2nd Edition
    Given:
        gamma -- the 3x3 rotation matrix
    Compute:
        Gamma -- the 12x12 transformation matrix
    """
    Gamma = np.zeros((12, 12))
    Gamma[0:3, 0:3] = gamma
    Gamma[3:6, 3:6] = gamma
    Gamma[6:9, 6:9] = gamma
    Gamma[9:12, 9:12] = gamma
    return Gamma



def global_stiffness_mat(members,node_cordinates, member_localz):
    K_global = np.zeros([len(node_cordinates)*6,len(node_cordinates)*6])
    for i in range(len(members)):
        E = members[i,0]
        nu = members[i,1]
        A = members[i,2]
        first_node = int(members[i,7])
        second_node = int(members[i,8])
        
        fnc = node_cordinates[first_node-1]  #first node cordinates
        snc = node_cordinates[second_node-1] #second node cordinates
        L = np.sqrt((fnc[0,0]-snc[0,0])**2 +   #finds the length of the member based on two nodes it connects
                    (fnc[0,1] - snc[0,1])**2 + 
                    (fnc[0,2] - snc[0,2])**2)
        Iy = members[i,3]
        Iz = members[i,4]
        J = members[i,6]   
        
        if type(member_localz[i]) == list:
            gamma = rotation_matrix_3D(fnc[0,0], fnc[0,1], fnc[0,2], snc[0,0], snc[0,1], snc[0,2], member_localz[i])
        else:
            gamma = rotation_matrix_3D(fnc[0,0], fnc[0,1], fnc[0,2], snc[0,0], snc[0,1], snc[0,2])
        
        Gamma = transformation_matrix_3D(gamma)
        
        
        # Compute local stiffness matrix
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        
        # Transform to global coordinate system
        k_global = Gamma.T @ k_local @ Gamma  # Transformation applied
        
        t_left = k_global[0:6,0:6]
        t_right = k_global[0:6,6:12]
        b_left = k_global[6:12,0:6]
        b_right = k_global[6:12,6:12]
    
        
        K_global[(first_node-1)*6:(first_node)*6, (first_node-1)*6:(first_node)*6] += t_left
        K_global[(second_node-1)*6:(second_node)*6, (second_node-1)*6:(second_node)*6] += b_right
        K_global[(first_node-1)*6:(first_node)*6, (second_node-1)*6:(second_node)*6] += t_right
        K_global[(second_node-1)*6:(second_node)*6, (first_node-1)*6:(first_node)*6] += b_left
      
   
    return K_global
    




def is_symmetric(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)



def solve_for_displacements_and_reactions(K_global, F_global, node_DOF):
    """
    Solves for displacements and reaction forces in a structure.
    
    Arguments:
    K_global -- Global stiffness matrix (size: [n_nodes*6, n_nodes*6])
    F_global -- Global force vector (size: [n_nodes*6, 1])
    node_DOF -- Nodal degrees of freedom (size: [n_nodes, 6], 0 = constrained, 1 = free)
    
    Returns:
    displacements -- Displacements at each node (size: [n_nodes*6, 1])
    reaction_forces -- Reaction forces at the constrained nodes (size: [n_constrained_nodes*6, 1])
    """
    
    # Step 1: Apply boundary conditions and reduce the stiffness matrix and force vector
    free_dofs = []
    constrained_dofs = []
    
    # Identify free and constrained DOFs
    for i in range(len(node_DOF)):
        for j in range(6):  # 6 DOFs per node
            if node_DOF[i, j] == 1:
                free_dofs.append(i * 6 + j)  # free DOF for node i
            else:
                constrained_dofs.append(i * 6 + j)  # constrained DOF for node i
    
    # Step 2: Reduce the system
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]  # Stiffness matrix for free DOFs
    F_reduced = F_global[free_dofs]  # Force vector for free DOFs
    
    # Step 3: Solve for displacements
    displacements_free = np.linalg.solve(K_reduced, F_reduced)  # Solve for displacements at free DOFs
    
    # Step 4: Assemble the full displacement vector (including constrained DOFs)
    displacements = np.zeros((K_global.shape[0], 1))  # Initialize displacement vector
    displacements[free_dofs] = displacements_free  # Assign displacements to free DOFs
    
    # Step 5: Calculate the reaction forces (forces at constrained DOFs)
    reaction_forces = np.dot(K_global[constrained_dofs, :], displacements)  # Reaction forces at constrained DOFs
    
    return displacements, reaction_forces, K_reduced





def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
    """
    local element geometric stiffness matrix
    source: p. 258 of McGuire's Matrix Structural Analysis 2nd Edition
    Given:
        material and geometric parameters:
            L, A, I_rho (polar moment of inertia)
        element forces and moments:
            Fx2, Mx2, My1, Mz1, My2, Mz2
    Context:
        load vector:
            [Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2]
        DOF vector:
            [u1, v1, w1, th_x1, th_y1, th_z1, u2, v2, w2, th_x2, th_y2, th_z2]
        Equation:
            [load vector] = [stiffness matrix] @ [DOF vector]
    Returns:
        12 x 12 geometric stiffness matrix k_g
    """
    k_g = np.zeros((12, 12))
    # upper triangle off diagonal terms
    k_g[0, 6] = -Fx2 / L
    k_g[1, 3] = My1 / L
    k_g[1, 4] = Mx2 / L
    k_g[1, 5] = Fx2 / 10.0
    k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
    k_g[1, 9] = My2 / L
    k_g[1, 10] = -Mx2 / L
    k_g[1, 11] = Fx2 / 10.0
    k_g[2, 3] = Mz1 / L
    k_g[2, 4] = -Fx2 / 10.0
    k_g[2, 5] = Mx2 / L
    k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
    k_g[2, 9] = Mz2 / L
    k_g[2, 10] = -Fx2 / 10.0
    k_g[2, 11] = -Mx2 / L
    k_g[3, 4] = -1.0 * (2.0 * Mz1 - Mz2) / 6.0
    k_g[3, 5] = (2.0 * My1 - My2) / 6.0
    k_g[3, 7] = -My1 / L
    k_g[3, 8] = -Mz1 / L
    k_g[3, 9] = -Fx2 * I_rho / (A * L)
    k_g[3, 10] = -1.0 * (Mz1 + Mz2) / 6.0
    k_g[3, 11] = (My1 + My2) / 6.0
    k_g[4, 7] = -Mx2 / L
    k_g[4, 8] = Fx2 / 10.0
    k_g[4, 9] = -1.0 * (Mz1 + Mz2) / 6.0
    k_g[4, 10] = -Fx2 * L / 30.0
    k_g[4, 11] = Mx2 / 2.0
    k_g[5, 7] = -Fx2 / 10.0
    k_g[5, 8] = -Mx2 / L
    k_g[5, 9] = (My1 + My2) / 6.0
    k_g[5, 10] = -Mx2 / 2.0
    k_g[5, 11] = -Fx2 * L / 30.0
    k_g[7, 9] = -My2 / L
    k_g[7, 10] = Mx2 / L
    k_g[7, 11] = -Fx2 / 10.0
    k_g[8, 9] = -Mz2 / L
    k_g[8, 10] = Fx2 / 10.0
    k_g[8, 11] = Mx2 / L
    k_g[9, 10] = (Mz1 - 2.0 * Mz2) / 6.0
    k_g[9, 11] = -1.0 * (My1 - 2.0 * My2) / 6.0
    # add in the symmetric lower triangle
    k_g = k_g + k_g.transpose()
    # add diagonal terms
    k_g[0, 0] = Fx2 / L
    k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
    k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
    k_g[3, 3] = Fx2 * I_rho / (A * L)
    k_g[4, 4] = 2.0 * Fx2 * L / 15.0
    k_g[5, 5] = 2.0 * Fx2 * L / 15.0
    k_g[6, 6] = Fx2 / L
    k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
    k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
    k_g[9, 9] = Fx2 * I_rho / (A * L)
    k_g[10, 10] = 2.0 * Fx2 * L / 15.0
    k_g[11, 11] = 2.0 * Fx2 * L / 15.0
    return k_g


def compute_internal_forces(members, node_cordinates, member_localz, displacements):
    """
    Computes internal forces and moments in local coordinates for each member.
    
    Arguments:
    members -- Member properties and connectivity matrix.
    node_cordinates -- Coordinates of each node.
    member_localz -- Local z-axis reference vectors for each member.
    displacements -- Global displacement vector.
    
    Returns:
    internal_forces -- Internal force and moment vectors for each member in local coordinates.
    """
    internal_forces = []
    
    for i in range(len(members)):
        E = members[i, 0]
        nu = members[i, 1]
        A = members[i, 2]
        first_node = int(members[i, 7])
        second_node = int(members[i, 8])
        
        fnc = node_cordinates[first_node-1]
        snc = node_cordinates[second_node-1]
        L = np.sqrt((fnc[0,0] - snc[0,0])**2 + 
                    (fnc[0,1] - snc[0,1])**2 + 
                    (fnc[0,2] - snc[0,2])**2)
        Iy = members[i, 3]
        Iz = members[i, 4]
        J = members[i, 6]
        
        if type(member_localz[i]) == list:
            gamma = rotation_matrix_3D(fnc[0,0], fnc[0,1], fnc[0,2], 
                                       snc[0,0], snc[0,1], snc[0,2], 
                                       member_localz[i])
        else:
            gamma = rotation_matrix_3D(fnc[0,0], fnc[0,1], fnc[0,2], 
                                       snc[0,0], snc[0,1], snc[0,2])
        
        Gamma = transformation_matrix_3D(gamma)   #gives 12x12 large Gamma transformation matrix
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
       
        # Extract nodal displacements for this member
        #d_global = displacements(
        d_global = np.vstack((displacements[(first_node-1)*6:(first_node)*6],
                               displacements[(second_node-1)*6:(second_node)*6]))
       
        # Transform global displacements to local coordinate system
        d_local = Gamma @ d_global  # Ensure it is a column vector

        
        # Compute internal force and moment vector in local coordinates
        f_local = k_local @ d_local
        
        internal_forces.append(f_local)
    
    return internal_forces




def global_3D_stiffness_mat(members,node_cordinates, member_localz, internal_forces):
    Kg_global = np.zeros([len(node_cordinates)*6,len(node_cordinates)*6])
    for i in range(len(members)):
        E = members[i,0]
        nu = members[i,1]
        A = members[i,2]
        first_node = int(members[i,7])
        second_node = int(members[i,8])
        
        fnc = node_cordinates[first_node-1]  #first node cordinates
        snc = node_cordinates[second_node-1] #second node cordinates
        L = np.sqrt((fnc[0,0]-snc[0,0])**2 +   #finds the length of the member based on two nodes it connects
                    (fnc[0,1] - snc[0,1])**2 + 
                    (fnc[0,2] - snc[0,2])**2)
        Iy = members[i,3]
        Iz = members[i,4]
        Irho = members[i,5]
        J = members[i,6]   
        
        Fx2 = internal_forces[i,6]
        Mx2 = internal_forces[i,9]
        My1 = internal_forces[i,4]
        Mz1 = internal_forces[i,5]
        My2 = internal_forces[i,10]
        Mz2 = internal_forces[i,11]
        
        
        if type(member_localz[i]) == list:
            gamma = rotation_matrix_3D(fnc[0,0], fnc[0,1], fnc[0,2], snc[0,0], snc[0,1], snc[0,2], member_localz[i])
        else:
            gamma = rotation_matrix_3D(fnc[0,0], fnc[0,1], fnc[0,2], snc[0,0], snc[0,1], snc[0,2])
        
        Gamma = transformation_matrix_3D(gamma)
        
        
        # Compute local 3d stiffness matrix
        kg_local = local_geometric_stiffness_matrix_3D_beam(L, A, Irho, Fx2, Mx2, My1, Mz1, My2, Mz2)
        
        # Transform to global coordinate system
        kg_global = Gamma.T @ kg_local @ Gamma  # Transformation applied
        
        t_left = kg_global[0:6,0:6]
        t_right = kg_global[0:6,6:12]
        b_left = kg_global[6:12,0:6]
        b_right = kg_global[6:12,6:12]
    
        
        Kg_global[(first_node-1)*6:(first_node)*6, (first_node-1)*6:(first_node)*6] += t_left
        Kg_global[(second_node-1)*6:(second_node)*6, (second_node-1)*6:(second_node)*6] += b_right
        Kg_global[(first_node-1)*6:(first_node)*6, (second_node-1)*6:(second_node)*6] += t_right
        Kg_global[(second_node-1)*6:(second_node)*6, (first_node-1)*6:(first_node)*6] += b_left
      
   
    return Kg_global
 

def solve_for_Kg_global_ff(Kg_global, F_global, node_DOF):
    
    
    # Step 1: Apply boundary conditions and reduce the stiffness matrix and force vector
    free_dofs = []
    constrained_dofs = []
    
    # Identify free and constrained DOFs
    for i in range(len(node_DOF)):
        for j in range(6):  # 6 DOFs per node
            if node_DOF[i, j] == 1:
                free_dofs.append(i * 6 + j)  # free DOF for node i
            else:
                constrained_dofs.append(i * 6 + j)  # constrained DOF for node i
    
    # Step 2: Reduce the system
    Kg_reduced = Kg_global[np.ix_(free_dofs, free_dofs)]  # Stiffness matrix for free DOFs
    
    return Kg_reduced



def solve_gen_eig(K_reduced,Kg_reduced):    #### needs to be free degree of freedom versions of those matrices
    eigenvalues, eigenvectors = sc.linalg.eig(K_reduced, -Kg_reduced)
    return eigenvalues, eigenvectors
