#! /usr/bin/env python

import numpy as np

def computeForces(vertices, fraction, force):
    vertices= np.array(vertices)
    k = int(fraction * 0.5 * len(vertices))

    ind_max = np.argpartition(+vertices[:,0], -k)[-k:]
    ind_min = np.argpartition(-vertices[:,0], -k)[-k:]

    forces = np.zeros((len(vertices), 3))

    forces[ind_max,0] = +force
    forces[ind_min,0] = -force
    return forces

def computeForcesFromFile(meshFile, fraction, force):
    import trimesh
    mesh = trimesh.load_mesh(meshFile)
    return computeForces(mesh.vertices, fraction, force)
    

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str)
    parser.add_argument('--fraction', type=float)
    parser.add_argument('--force', type=float)
    parser.add_argument('--out', type=str, default="stdout")
    args = parser.parse_args()

    forces = computeForcesFromFile(args.mesh, args.fraction, args.force)

    output = args.out
    if output == "stdout":
        output = sys.stdout
    
    np.savetxt(output, forces)
