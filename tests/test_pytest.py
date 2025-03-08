import pytest
import numpy as np
import math as mt
from direct_stiffness import (
    global_stiffness_mat, solve_for_displacements_and_reactions, compute_internal_forces,
    global_3D_stiffness_mat, solve_for_Kg_global_ff, solve_gen_eig
)

def test_global_stiffness_mat():
    # Define test inputs
    members = np.matrix([[500, 0.3, np.pi*0.5**2, (np.pi*0.5**4)/4, (np.pi*0.5**4)/4, (np.pi*0.5**4)/2, (np.pi*0.5**4)/2, 1, 5]])
    node_cordinates = np.matrix([[0,0,0],[10,0,0]])
    member_localz = np.array([0], dtype=object)
    
    # Run function
    K_global = global_stiffness_mat(members, node_cordinates, member_localz)
    
    # Assert conditions
    assert K_global.shape == (12, 12), "Global stiffness matrix size incorrect"

def test_solve_for_displacements_and_reactions():
    K_global = np.eye(12)  # Identity for test
    F_global = np.ones((12,1))
    node_DOF = np.ones((2,6))
    
    displacements, reaction_forces, K_reduced = solve_for_displacements_and_reactions(K_global, F_global, node_DOF)
    
    assert displacements.shape == (12,1), "Displacement vector size incorrect"
    assert reaction_forces.shape == (12,1), "Reaction forces vector size incorrect"

def test_compute_internal_forces():
    members = np.matrix([[500, 0.3, np.pi*0.5**2, (np.pi*0.5**4)/4, (np.pi*0.5**4)/4, (np.pi*0.5**4)/2, (np.pi*0.5**4)/2, 1, 5]])
    node_cordinates = np.matrix([[0,0,0],[10,0,0]])
    member_localz = np.array([0], dtype=object)
    displacements = np.zeros((12,1))
    
    internal_forces = compute_internal_forces(members, node_cordinates, member_localz, displacements)
    
    assert len(internal_forces) == len(members), "Internal forces computation incorrect"

def test_global_3D_stiffness_mat():
    members = np.matrix([[500, 0.3, np.pi*0.5**2, (np.pi*0.5**4)/4, (np.pi*0.5**4)/4, (np.pi*0.5**4)/2, (np.pi*0.5**4)/2, 1, 5]])
    node_cordinates = np.matrix([[0,0,0],[10,0,0]])
    member_localz = np.array([0], dtype=object)
    internal_forces = np.zeros(len(members))
    
    Kg_global = global_3D_stiffness_mat(members, node_cordinates, member_localz, internal_forces)
    
    assert Kg_global.shape == (12, 12), "Geometric stiffness matrix size incorrect"

def test_solve_for_Kg_global_ff():
    Kg_global = np.eye(12)
    F_global = np.ones((12,1))
    node_DOF = np.ones((2,6))
    
    Kg_reduced = solve_for_Kg_global_ff(Kg_global, F_global, node_DOF)
    
    assert Kg_reduced.shape[0] <= 12, "Reduced geometric stiffness matrix size incorrect"

def test_solve_gen_eig():
    K_global = np.eye(12)
    Kg_global = np.eye(12) * 2
    
    eigenvalues, eigenvectors = solve_gen_eig(K_global, Kg_global)
    
    assert len(eigenvalues) > 0, "Eigenvalue computation failed"
    assert eigenvectors.shape[0] == 12, "Eigenvector size incorrect"
    assert min(eigenvalues) > 0, "Critical force calculation incorrect"
