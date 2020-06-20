#!/usr/bin/env python

import numpy as np

def rotate_coords(vertices):
    vertices -= np.mean(vertices, axis=0)
    cov = np.cov(vertices.T)

    eig_vals, eig_vecs = np.linalg.eig(cov)    
    
    vertices = vertices * np.matrix(eig_vecs)
    cov = np.cov(vertices.T)

    return np.array(vertices)
    
def compute_diameters_from_vertices(vertices):
    vertices = np.array(vertices)
    vertices = rotate_coords(vertices)
    
    box_lo = np.amin(vertices, axis=0)
    box_hi = np.amax(vertices, axis=0)

    extents = box_hi - box_lo
    extents = np.sort(extents)[::-1]
    
    Dlong  = extents[0]
    Dshort = extents[1]
    return Dlong, Dshort

def compute_diameters(filename):
    import trimesh
    mesh = trimesh.load(filename)
    return compute_diameters_from_vertices(mesh.vertices)

def process(mesh_files, eq_mesh, transformation):
    num_files = len(mesh_files)
    quants = np.zeros(num_files)

    (L0, W0) = compute_diameters(eq_mesh)
    
    for i, fname in enumerate(mesh_files):
        (L, W) = compute_diameters(fname)
        quants[i] = transformation(L, W, L0)

    return quants

def transformation_LW_ratio (L, W, L0): return L / W
def transformation_LL0_ratio(L, W, L0): return L / L0

def process_LW (mesh_files, eq_mesh): return process(mesh_files, eq_mesh, transformation_LW_ratio)
def process_LL0(mesh_files, eq_mesh): return process(mesh_files, eq_mesh, transformation_LL0_ratio)

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_files', type=str, nargs='+', required=True, help="mesh files (e.g. ply)")
    parser.add_argument('--eq_mesh', type=str, required=True, help="equilibrium mesh")
    parser.add_argument('--transform', type=str, choices=["LW", "LL0"], default="LW", help="transformation to be performed from the diameters")
    parser.add_argument('--out', type=str, default="stdout", help="output file")
    args = parser.parse_args()

    mesh_files = args.mesh_files
    eq_mesh    = args.eq_mesh

    if args.transform == "LW":
        quants = process_LW(mesh_files, eq_mesh)
    else:
        quants = process_LL0(mesh_files, eq_mesh)

    output = args.out
    if output == "stdout":
        output = sys.stdout
    
    np.savetxt(output, quants)
