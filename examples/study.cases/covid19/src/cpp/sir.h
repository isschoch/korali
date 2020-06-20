#pragma once

#include <array>
#include <vector>

std::vector<double> sir_infected_so_far(double S0, double I0, double R0,
                                        int days, long long N, double beta,
                                        double gamma, double tint, double dint,
                                        double kbeta);
