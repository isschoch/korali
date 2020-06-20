#! /bin/bash -f

module load cray-python/3.6.5.6
module load cray-python/2.7.15.7

rr=results.txt 

NPROC=12

sigmalj=`sed '1q;d' params.txt`
epsilonlj=`sed '2q;d' params.txt`
echo "CG Water simulation with: sigma = $sigmalj e = $epsilonlj" > run.log

# mapping
M=1.0     #mapping
nrepeat=8	# number of particles in box nrepeat*nrepeat*nrepeat

#write configuration
python2 ./writeInitEqu.py "$M " "$nrepeat"

# Thermodynamic conditions
T=298.0
DensSet=0.997

# Run LAMMPS @ T
srun --exclusive -N 1 -n $NPROC lmp_mpi -var Tset $T -var DensSet $DensSet -var M $M -var sigma $sigmalj -var epsilon $epsilonlj -in in.LAMMPSequ.txt >> run.log

trj=trjNPT.txt

# Check RDF, dens, rewrite or enlarge
outCheck="$(python2 ./writeInit.py "$M " "$trj")"

if [[ $outCheck == -1 ]]; then
 echo "Density out" >> run.log
 echo "298.0   -INFINITY" > $rr
else
 srun --exclusive -N 1 -n $NPROC lmp_mpi -var Tset $T -var DensSet $DensSet -var M $M -var sigma $sigmalj -var epsilon $epsilonlj -in in.LAMMPS.txt > lmp_output.txt

 # Value @ T
 out="$(python2 ./compute_density.py "$T")"
 echo "Value at T=298.0: $out" >> run.log
 echo "298.0   $out" > $rr
fi



