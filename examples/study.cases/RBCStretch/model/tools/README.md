# Tools

RBC stretching tools.  
run tests with:  

	mir.make test

## diameters.py

compute principal diameters by first rotating a given mesh (vertices) and then computing the extents (bounding box)

## stretchForce

create array of forces per vertex given a mesh.
This corresponds to stretching forces in opposite directions
Parameters:

	fraction: number of vertices to which the force is applied divided by total numer of vertices
	force: force magnitude

