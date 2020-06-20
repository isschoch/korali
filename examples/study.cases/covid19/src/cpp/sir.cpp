#include "sir.h"
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>

using State = boost::array<double, 3>;
using Stepper = boost::numeric::odeint::runge_kutta_dopri5<State>;

double smooth_trans(double u0, double u1, double t, double tc, double teps)
{
  const double t0 = tc - teps;
  const double t1 = tc + teps;
  return t <= t0 ? u0 : t >= t1 ? u1 : u0 + (u1 - u0) * (1 - std::cos(M_PI / (t1 - t0) * (t - t0))) * 0.5;
}

double linear_trans(double u0, double u1, double t, double tc, double teps)
{
  const double t0 = tc - teps;
  const double t1 = tc + teps;
  return t <= t0 ? u0 : t >= t1 ? u1 : u0 + (u1 - u0) * (t - t0) / (t1 - t0);
}

static auto make_sir_rhs(double N, double beta, double gamma, double tint, double dint, double kbeta)
{
  /// RHS of the SIR model. We preprocess beta/N for efficiency.
  return [beta_over_N = beta / N, gamma, tint, dint, kbeta](const State &x, State &dxdt, double t) {
    double S = x[0];
    double I = x[1];
    double R = x[2];

    //auto trans = smooth_trans;
    auto trans = linear_trans;
    const double b_N = beta_over_N * trans(1., kbeta, t, tint, dint * 0.5);
    double A = b_N * S * I;
    double B = gamma * I;
    dxdt[0] = -A;
    dxdt[1] = A - B;
    dxdt[2] = B;
  };
}

/**
    Arguments:
        S0, I0, R0: Initial state (at day=0)
        days: Number of days to evaluate.
        N: Population
        beta, gamma: SIR parameters

    Return value:
        Array {N - S(t) | t = 1, ..., days}
  */
std::vector<double>
sir_infected_so_far(double S0, double I0, double R0, int days, long long N, double beta, double gamma, double tint, double dint, double kbeta)
{
  const int steps_per_day = 10;
  const double dt = 1.0 / steps_per_day;

  auto rhs = make_sir_rhs(N, beta, gamma, tint, dint, kbeta);

  std::vector<double> result;
  result.reserve(days);

  // Observer gets called for each time step evaluated by the integrator.
  // We consider only every `steps_per_day` steps
  // (skipping the first one as well), and store N - S(t).
  auto observer = [&result, N, cnt = 0](State y, double t) mutable {
    if (cnt % steps_per_day == 0 && cnt > 0)
    {
      // printf("t=%g  y=%g %g %g\n", t, y[0], y[1], y[2]);
      result.push_back(N - y[0]);
    }
    ++cnt;
  };

  State y0{S0, I0, R0};
  boost::numeric::odeint::integrate_n_steps(
    Stepper{}, rhs, y0, 0.0, dt, days * steps_per_day, observer);
  return result;
}
