/**********************************************************************/
// A generic stencil-based 3D Jacobi solver                           //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin                                             //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include "mpi.h"
#include "korali.hpp"
#include <vector>

std::vector<double> getPointData();
std::vector<double> getReferenceData();
void jacobi(korali::Sample& k);
