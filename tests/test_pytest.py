import pytest
import numpy as np
import math as mt
from Scripts import direct_stiffness as ds

node_cordinates = np.matrix([[0,0,0],
                            [10,0,0],
                            [10,20,0],
                            [0,20,0],
                            [0,0,25],
                            [10,0,25],
                            [10,20,25],
                            [0,20,25]])

#enter nodal forces/torques for each node as a row (Fx, Fy, Fx, Mx, My, Mz)
#the first row will be considered node 1 and so on
node_force = np.matrix([[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                       [0,0,0,0,0,0],
                       [0,0,0,0,0,0],
                       [0,0,-1,0,0,0],
                       [0,0,-1,0,0,0],
                       [0,0,-1,0,0,0],
                       [0,0,-1,0,0,0]])

#enter nodal degrees of freedom for each node as a row (x,y,z,rot_x,rot_y,rot_z)
#0 for constrained and 1 for free
#the first row will be considered node 1 and so on
node_DOF = np.matrix([[0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [1,1,1,1,1,1],
                     [1,1,1,1,1,1],
                     [1,1,1,1,1,1],
                     [1,1,1,1,1,1]])

member_localz = np.array([0,0,0,0,0,0,0,0], dtype=object)


#enter member properties and connecting nodes E, nu, A, Iy, Iz, J, 1, 2
#the first row will be considered member A, the second row will be member B and so on
r = .5
E = 500
nu = .3
A = np.pi*r**2
Iy = (np.pi*r**4)/4
Iz = (np.pi*r**4)/4
Irho = (np.pi*r**4)/2
J = (np.pi*r**4)/2





members = np.matrix([[E, nu, A, Iy, Iz, Irho, J, 1, 5],    #member E0
                     [E, nu, A, Iy, Iz, Irho, J, 2, 6],    #member E1
                     [E, nu, A, Iy, Iz, Irho, J, 3, 7],    #member E2
                     [E, nu, A, Iy, Iz, Irho, J, 4, 8],    #member E3
                     [E, nu, A, Iy, Iz, Irho, J, 5, 6],     #member E4
                     [E, nu, A, Iy, Iz, Irho, J, 6, 7],     #member E5
                     [E, nu, A, Iy, Iz, Irho, J, 7, 8],     #member E6
                     [E, nu, A, Iy, Iz, Irho, J, 8, 5]])     #member E7



F_global = np.zeros((len(node_DOF)*6, 1))  # 24x1 zero matrix    
DOF_global = np.zeros((len(node_DOF)*6, 1))  # 24x1 zero matrix

for i in range(len(node_force)):  
    F_global[i*6:i*6+6, 0] = node_force[i].flatten()

for i in range(len(node_DOF)):  
    DOF_global[i*6:i*6+6, 0] = node_DOF[i].flatten()

K_global = ds.global_stiffness_mat(members, node_cordinates, member_localz)

displacements, reaction_forces, K_reduced = ds.solve_for_displacements_and_reactions(K_global, F_global, node_DOF)


def test_global_stiffness_mat():
    
    # Run function
    K_global = ds.global_stiffness_mat(members, node_cordinates, member_localz)
    
    # Assert conditions
    assert K_global.shape == (48, 48), "Global stiffness matrix size incorrect"

def test_solve_for_displacements_and_reactions():
    
    
    displacements, reaction_forces, K_reduced = ds.solve_for_displacements_and_reactions(K_global, F_global, node_DOF)
    
    assert displacements.shape == (48,1), "Displacement vector size incorrect"
    assert reaction_forces.shape == (24,1), "Reaction forces vector size incorrect"
    assert K_reduced.shape == (24,24), "K_reduced mat size incorrect"



def test_compute_internal_forces():
    
    internal_forces = ds.compute_internal_forces(members, node_cordinates, member_localz, displacements)
    
    assert len(internal_forces) == len(members), "Internal forces computation incorrect"


def test_global_3D_stiffness_mat():
    
    internal_forces = np.array(ds.compute_internal_forces(members, node_cordinates, member_localz, displacements))
    Kg_global = ds.global_3D_stiffness_mat(members, node_cordinates, member_localz, internal_forces)
    
    assert Kg_global.shape == (48, 48), "Geometric stiffness matrix size incorrect"



def test_solve_for_Kg_global_ff():
    internal_forces = np.array(ds.compute_internal_forces(members, node_cordinates, member_localz, displacements))
    Kg_global = ds.global_3D_stiffness_mat(members, node_cordinates, member_localz, internal_forces)
    
    Kg_reduced = ds.solve_for_Kg_global_ff(Kg_global, F_global, node_DOF)
    
    assert Kg_reduced.shape == (24,24), "Reduced geometric stiffness matrix size incorrect"

def test_solve_gen_eig():
    K_reduced = np.eye(12)
    Kg_reduced = np.eye(12)
    
    eigenvalues, eigenvectors = ds.solve_gen_eig(K_reduced, Kg_reduced)
    
    assert eigenvalues[1] == (-1+0j), "Eigenvalue computation failed"
    assert eigenvectors.shape == (12,12), "Eigenvector size incorrect"
    
