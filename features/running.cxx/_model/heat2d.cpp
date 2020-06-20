/**********************************************************************/
// A simple Multigrid Solver for the Heat Equation                    //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin, Georgios Arampatzis                        //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include "heat2d.h"
#include "string.h"
#include <limits>
#include <math.h>
#include <stdio.h>

pointsInfo __p;

void heat2DSolver(korali::Sample &k)
{
  double tolerance = 1e-8; // L2 Difference Tolerance before reaching convergence.
  size_t N0 = 7;           // 2^N0 + 1 elements per side

  // Multigrid parameters -- Find the best configuration!
  int gridCount = 6;       // Number of Multigrid levels to use
  int downRelaxations = 4; // Number of Relaxations before restriction
  int upRelaxations = 1;   // Number of Relaxations after prolongation

  gridLevel *g = generateInitialConditions(N0, gridCount, k["Parameters"]);

  while (g[0].L2NormDiff > tolerance) // Multigrid solver start
  {
    applyGaussSeidel(g, 0, downRelaxations); // Relaxing the finest grid first
    calculateResidual(g, 0);                 // Calculating Initial Residual

    for (int grid = 1; grid < gridCount; grid++) // Going down the V-Cycle
    {
      applyRestriction(g, grid);                  // Restricting the residual to the coarser grid's solution vector (f)
      applyGaussSeidel(g, grid, downRelaxations); // Smoothing coarser level
      calculateResidual(g, grid);                 // Calculating Coarse Grid Residual
    }

    for (int grid = gridCount - 1; grid > 0; grid--) // Going up the V-Cycle
    {
      applyProlongation(g, grid);               // Prolonging solution for coarser level up to finer level
      applyGaussSeidel(g, grid, upRelaxations); // Smoothing finer level
    }

    calculateL2Norm(g, 0); // Calculating Residual L2 Norm
  }                        // Multigrid solver end

  // Saving the value of temperatures at specified points
  double h = 1.0 / (g[0].N - 1);
  double sig = k["Parameters"][3];
  for (size_t i = 0; i < __p.refTemp.size(); i++)
  {
    int m = ceil(__p.xPos[i] / h);
    int l = ceil(__p.yPos[i] / h);
    k["Reference Evaluations"][i] = g[0].U[m][l];
    k["Standard Deviation"][i] = sig;
  }

  freeGrids(g, gridCount);
}

void applyGaussSeidel(gridLevel *g, int l, int relaxations)
{
  double h2 = g[l].h * g[l].h;
  for (int r = 0; r < relaxations; r++)
  {
    for (size_t j = 1; j < g[l].N - 1; j++) // Gauss-Seidel Iteration -- Credit: Claudio Cannizzaro
      for (size_t i = 1; i < g[l].N - 1; i++)
        g[l].U[i][j] = ((g[l].U[i - 1][j] + g[l].U[i + 1][j] + g[l].U[i][j - 1] + g[l].U[i][j + 1]) + g[l].f[i][j] * h2) * 0.25;
  }
}

void calculateResidual(gridLevel *g, int l)
{
  double h2 = 1.0 / pow(g[l].h, 2);

  for (size_t i = 1; i < g[l].N - 1; i++)
    for (size_t j = 1; j < g[l].N - 1; j++)
      g[l].Res[i][j] = g[l].f[i][j] + (g[l].U[i - 1][j] + g[l].U[i + 1][j] - 4 * g[l].U[i][j] + g[l].U[i][j - 1] + g[l].U[i][j + 1]) * h2;
}

void calculateL2Norm(gridLevel *g, int l)
{
  double tmp = 0.0;

  for (size_t i = 0; i < g[l].N; i++)
    for (size_t j = 0; j < g[l].N; j++)
      g[l].Res[i][j] = g[l].Res[i][j] * g[l].Res[i][j];

  for (size_t i = 0; i < g[l].N; i++)
    for (size_t j = 0; j < g[l].N; j++)
      tmp += g[l].Res[i][j];

  g[l].L2Norm = sqrt(tmp);
  g[l].L2NormDiff = abs(g[l].L2NormPrev - g[l].L2Norm);
  g[l].L2NormPrev = g[l].L2Norm;
}

void applyRestriction(gridLevel *g, int l)
{
  for (size_t i = 1; i < g[l].N - 1; i++)
    for (size_t j = 1; j < g[l].N - 1; j++)
      g[l].f[i][j] = (1.0 * (g[l - 1].Res[2 * i - 1][2 * j - 1] + g[l - 1].Res[2 * i - 1][2 * j + 1] + g[l - 1].Res[2 * i + 1][2 * j - 1] + g[l - 1].Res[2 * i + 1][2 * j + 1]) +
                      2.0 * (g[l - 1].Res[2 * i - 1][2 * j] + g[l - 1].Res[2 * i][2 * j - 1] + g[l - 1].Res[2 * i + 1][2 * j] + g[l - 1].Res[2 * i][2 * j + 1]) +
                      4.0 * (g[l - 1].Res[2 * i][2 * j])) *
                     0.0625;

  for (size_t i = 0; i < g[l].N; i++)
    for (size_t j = 0; j < g[l].N; j++) // Resetting U vector for the coarser level before smoothing -- Find out if this is really necessary.
      g[l].U[i][j] = 0;
}

void applyProlongation(gridLevel *g, int l)
{
  for (size_t i = 1; i < g[l].N - 1; i++)
    for (size_t j = 1; j < g[l].N - 1; j++)
      g[l - 1].U[2 * i][2 * j] += g[l].U[i][j];

  for (size_t i = 1; i < g[l].N; i++)
    for (size_t j = 1; j < g[l].N - 1; j++)
      g[l - 1].U[2 * i - 1][2 * j] += (g[l].U[i - 1][j] + g[l].U[i][j]) * 0.5;

  for (size_t i = 1; i < g[l].N - 1; i++)
    for (size_t j = 1; j < g[l].N; j++)
      g[l - 1].U[2 * i][2 * j - 1] += (g[l].U[i][j - 1] + g[l].U[i][j]) * 0.5;

  for (size_t i = 1; i < g[l].N; i++)
    for (size_t j = 1; j < g[l].N; j++)
      g[l - 1].U[2 * i - 1][2 * j - 1] += (g[l].U[i - 1][j - 1] + g[l].U[i - 1][j] + g[l].U[i][j - 1] + g[l].U[i][j]) * 0.25;
}

gridLevel *generateInitialConditions(size_t N0, int gridCount, std::vector<double> pars)
{
  // Problem Parameters
  double intensity = pars[0];
  double xPos = pars[1];
  double yPos = pars[2];
  double width = 0.05;

  // Allocating Grids
  gridLevel *g = (gridLevel *)calloc(sizeof(gridLevel), gridCount);
  for (int i = 0; i < gridCount; i++)
  {
    g[i].N = pow(2, N0 - i) + 1;
    g[i].h = 1.0 / (g[i].N - 1);

    g[i].U = (double **)calloc(sizeof(double *), g[i].N);
    for (size_t j = 0; j < g[i].N; j++) g[i].U[j] = (double *)calloc(sizeof(double), g[i].N);
    g[i].Res = (double **)calloc(sizeof(double *), g[i].N);
    for (size_t j = 0; j < g[i].N; j++) g[i].Res[j] = (double *)calloc(sizeof(double), g[i].N);
    g[i].f = (double **)calloc(sizeof(double *), g[i].N);
    for (size_t j = 0; j < g[i].N; j++) g[i].f[j] = (double *)calloc(sizeof(double), g[i].N);

    g[i].L2Norm = 0.0;
    g[i].L2NormPrev = std::numeric_limits<double>::max();
    g[i].L2NormDiff = std::numeric_limits<double>::max();
  }

  // Initial Guess
  for (size_t i = 0; i < g[0].N; i++)
    for (size_t j = 0; j < g[0].N; j++) g[0].U[i][j] = 1.0;

  // Boundary Conditions
  for (size_t i = 0; i < g[0].N; i++) g[0].U[0][i] = 0.0;
  for (size_t i = 0; i < g[0].N; i++) g[0].U[g[0].N - 1][i] = 0.0;
  for (size_t i = 0; i < g[0].N; i++) g[0].U[i][0] = 0.0;
  for (size_t i = 0; i < g[0].N; i++) g[0].U[i][g[0].N - 1] = 0.0;

  // F
  for (size_t i = 0; i < g[0].N; i++)
    for (size_t j = 0; j < g[0].N; j++)
    {
      double h = 1.0 / (g[0].N - 1);
      double x = i * h;
      double y = j * h;

      g[0].f[i][j] = 0.0;

      // Heat Source: Candle
      g[0].f[i][j] += -(4 * intensity * exp(-(pow(xPos - y, 2) + pow(yPos - x, 2)) / width) * (pow(xPos, 2) - 2 * xPos * y + pow(yPos, 2) - 2 * yPos * x + pow(y, 2) + pow(x, 2) - width)) / pow(width, 2);
    }

  return g;
}

void freeGrids(gridLevel *g, int gridCount)
{
  for (int i = 0; i < gridCount; i++)
  {
    for (size_t j = 0; j < g[i].N; j++) free(g[i].U[j]);
    for (size_t j = 0; j < g[i].N; j++) free(g[i].f[j]);
    for (size_t j = 0; j < g[i].N; j++) free(g[i].Res[j]);
    free(g[i].U);
    free(g[i].f);
    free(g[i].Res);
  }
  free(g);
}

pointsInfoStruct &heat2DInit(int *argc, char **argv[])
{
  int problemNumber = 1;
  FILE *problemFile;

  size_t nPoints = 0;

  printf("Heat2D - Running problem from data.in... \n");
  problemFile = fopen("_model/data.in", "r");
  fscanf(problemFile, "%lu", &nPoints);

  __p.xPos.resize(nPoints);
  __p.yPos.resize(nPoints);
  __p.refTemp.resize(nPoints);

  for (size_t i = 0; i < nPoints; i++)
  {
    double val;
    fscanf(problemFile, "%le ", &val);
    __p.xPos[i] = val;
    fscanf(problemFile, "%le ", &val);
    __p.yPos[i] = val;
    fscanf(problemFile, "%le ", &val);
    __p.refTemp[i] = val;
  }

  return __p;
}
