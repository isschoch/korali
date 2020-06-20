#ifndef _DIRECT_HPP_
#define _DIRECT_HPP_

#include "korali.hpp"

void direct(korali::Sample &k)
{
  float x = k["Parameters"][0];
  k["F(x)"] = -0.5 * x * x;
}

#endif
