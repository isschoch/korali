#!/usr/bin/env python

import numpy as np
import sys, os
import korali

this_dir, this_filename = os.path.split(__file__)
data_dir = os.path.join(this_dir, "data")

mesh_fname = os.path.join(data_dir, "1986.scaled.off")
data_fname = os.path.join(data_dir, "Mills.2004.txt")

# Stretch Cell for given Parameters (x0, ks) and Force
def run_one_force(comm_address, force, x0, ks):

    import mirheo as mir
    import trimesh
    toolsPath = os.path.join(this_dir, "tools")
    #print(toolsPath)
    sys.path.append(toolsPath)
    import stretchForce as stretch
    import diameters
    from rbc_params import set_rbc_params
    from rbc_params import print_rbc_params

    def original_diameter(mesh):
        pos = np.array(mesh.vertices, copy=True)
        (D0, D1) = diameters.computeDiameters(pos)
        return 0.5 * (D0 + D1)

    tend      = 50.0
    dt        = 0.001
    
    mesh = trimesh.load_mesh(mesh_fname)
    
    safety = 3
    box_lo = np.amin(mesh.vertices, axis=0)
    box_hi = np.amax(mesh.vertices, axis=0)
    
    domain = (box_hi - box_lo) * safety + np.ones(3)
    domain = tuple(np.array(domain, dtype=int))
    ranks  = (1, 1, 1)
    
    u = mir.Mirheo(ranks, domain, dt, debug_level=0, log_filename='stderr', comm_ptr=comm_address, no_splash=True)
    
    mesh_rbc = mir.ParticleVectors.MembraneMesh(mesh.vertices.tolist(), mesh.faces.tolist())
    pv_rbc   = mir.ParticleVectors.MembraneVector("rbc", mass=1.0, mesh=mesh_rbc)
    ic_rbc   = mir.InitialConditions.Membrane([[domain[0] * 0.5,
                                                domain[1] * 0.5,
                                                domain[2] * 0.5,
                                                1.0, 0.0, 0.0, 0.0]])
    u.registerParticleVector(pv_rbc, ic_rbc)
    
    # mu_ph = 2.5 # [micro N / m]
    force_scale = 0.025 # [pN]
    force = force / force_scale
    D0    = original_diameter(mesh)
    D0_ph = 7.739 # [micro m]
    
    prms_rbc = set_rbc_params(mesh, x0, ks)
    prms_rbc["gammaT"] = 1.0 # for killing oscillations
    
    int_rbc = mir.Interactions.MembraneForces("int_rbc", **prms_rbc, stress_free=True)
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv_rbc)
    u.registerInteraction(int_rbc)
    u.setInteraction(int_rbc, pv_rbc, pv_rbc)
    
    fraction = 0.05 * 2
    n = int(len(mesh.vertices) * fraction)
    force = force / (0.5 * n)

    forces = stretch.computeForces(mesh.vertices, fraction, force).tolist()
    
    u.registerPlugins(mir.Plugins.createMembraneExtraForce("stretchForce", pv_rbc, forces))
    u.registerPlugins(mir.Plugins.createStats('stats', "stats.txt", 1000))
    
    u.run(int(tend/dt))

    Dshort = 0
    Dlong  = 0
    
    if u.isMasterTask():
        rbc_pos = pv_rbc.getCoordinates()
        (Dlong, Dshort) = diameters.computeDiameters(rbc_pos)
                    
        # make it dimensional again
        Dlong  = (Dlong  / D0) * D0_ph
        Dshort = (Dshort / D0) * D0_ph

    del u
    del vv
    del int_rbc
    del prms_rbc
    del mesh_rbc
    del pv_rbc
    del ic_rbc
    return Dshort, Dlong

# Evaluate Stretching Experiments
def evaluate( korali_obj, forces, comm_address, verbose = False ):
    x0 = korali_obj["Parameters"][0]
    ks = korali_obj["Parameters"][1]

    diam_lo_sim = np.zeros(len(forces))
    diam_hi_sim = np.zeros(len(forces))

    for i, f in enumerate(forces):
        if verbose:
            print("in: f {0}, x0 {1}, ks {2}".format(f, x0, ks))
            sys.stdout.flush()
        (dlo, dhi) = run_one_force(comm_address, f, x0, ks)
        if verbose:
            print("out: f {0}, dlo {1}, dhi {2}".format(f, dlo, dhi))
            sys.stdout.flush()

        diam_lo_sim[i] = dlo
        diam_hi_sim[i] = dhi

    result = []

    for i in range(len(diam_lo_sim)):
        result.append(diam_lo_sim[i])

    for i in range(len(diam_hi_sim)):
        result.append(diam_hi_sim[i])

    korali_obj["Reference Evaluations"] = result
