#! /usr/bin/env python

import numpy as np

def rotate_coords(vertices):
    vertices -= np.mean(vertices, axis=0)
    cov = np.cov(vertices.T)

    eig_vals, eig_vecs = np.linalg.eig(cov)    
    
    vertices = vertices * np.matrix(eig_vecs)
    cov = np.cov(vertices.T)

    return np.array(vertices)
    

def computeDiameters(vertices):
    vertices = np.array(vertices)
    vertices = rotate_coords(vertices)
    
    box_lo = np.amin(vertices, axis=0)
    box_hi = np.amax(vertices, axis=0)

    extents = box_hi - box_lo
    extents = np.sort(extents)[::-1]
    
    Dlong  = extents[0]
    Dshort = extents[1]
    return Dlong, Dshort
    

if __name__ == "__main__":
    import argparse, sys, trimesh

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--out', type=str, default="stdout")
    args = parser.parse_args()

    mesh = trimesh.load_mesh(args.mesh)
    (Dlong, Dshort) = computeDiameters(mesh.vertices)

    output = args.out
    if output == "stdout":
        output = sys.stdout
    
    np.savetxt(output, [Dlong, Dshort])
