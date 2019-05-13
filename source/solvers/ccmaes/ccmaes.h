#ifndef _KORALI_CCMAES_H_
#define _KORALI_CCMAES_H_

#include "solvers/base/base.h"
#include "variables/gaussian/gaussian.h"
#include <chrono>
#include <map>

namespace Korali::Solver
{

class CCMAES : public Korali::Solver::Base
{
 public:

 // Constructor / Destructor
 CCMAES(nlohmann::json& js);
 ~CCMAES();

 // Runtime Methods (to be inherited from base class in the future)
 void prepareGeneration();
 bool checkTermination();
 void updateDistribution(const double *fitnessVector);
 void run() override;
 void processSample(size_t sampleId, double fitness) override;

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
 void setState(nlohmann::json& js) override;

 private:

 // Korali Runtime Variables
 int _fitnessSign;
 std::string _objective;
 double* _fitnessVector; /* objective function values [_s] */
 double* _samplePopulation; /* sample coordinates [_s x _k->N] */
 size_t _currentGeneration; /* generation count */
 bool* _initializedSample;
 char _terminationReason[500];

 size_t _finishedSamples;
 size_t _s; /* number of samples per generation */
 size_t _mu; /* number of best samples for mean / cov update */
 std::string _muType; /* Linear, Equal or Logarithmic */
 double* _muWeights; /* weights for mu best samples */
 double _muEffective; /* variance effective selection mass */
 double _muCovariance;

 double _sigmaCumulationFactor; /* default calculated from muEffective and dimension */
 double _dampFactor; /* dampening parameter determines controls step size adaption */
 double _cumulativeCovariance; /* default calculated from dimension */
 double _covarianceMatrixLearningRate;
 double _chiN; /* expectation of ||N(0,I)|| */

 bool   _enablediag; /* enable diagonal covariance matrix */
 size_t _diagonalCovarianceMatrixEvalFrequency;
 size_t _covarianceEigenEvalFreq;

 // Stop conditions
 size_t _maxFitnessEvaluations;   // Defines maximum number of fitness evaluations
 double _stopFitnessDiffThreshold; // Defines minimum function value differences before stopping
 double _stopMinDeltaX; // Defines minimum delta of input parameters among generations before it stops.
 double _stopMinFitness; // Defines the minimum fitness allowed, otherwise it stops
 double _stopTolUpXFactor; // Defines the minimum fitness allowed, otherwise it stops
 double _stopCovCond; // Defines the maximal condition number of the covariance matrix
 size_t _maxGenenerations; // Max number of generations.
 std::string _ignorecriteria; /* Termination Criteria(s) to ignore:
    Fitness Value, Fitness Diff Threshold, Max Standard Deviation,
    Max Kondition Covariance, No Effect Axis, No Effect Standard Deviation,
    Max Model Evaluations, Max Generations */

 // Private CMAES-Specific Variables
 double sigma;  /* step size */
 Variable::Gaussian* _gaussianGenerator;

 double bestEver; /* best ever fitness */
 double prevBest; /* best ever fitness from previous generation */
 double *rgxmean; /* mean "parent" */
 double *rgxbestever; /* bestever vector */
 double *curBestVector; /* current best vector */
 size_t *index; /* sorting index of current sample pop (index[0] idx of current best). */
 double currentFunctionValue; /* best fitness current generation */
 double prevFunctionValue; /* best fitness previous generation */

 double **C; /* lower triangular matrix: i>=j for C[i][j] */
 double **B; /* matrix with eigenvectors in columns */
 double *rgD; /* axis lengths (sqrt(Evals)) */
 
 double **Z; /* randn() */
 double **BD; /* B*D */
 double **BDZ; /* B*D*randn() */

 double *rgpc; /* evolution path for cov update */
 double *rgps; /* conjugate evolution path for sigma update */
 double *rgxold; /* mean "parent" previous generation */
 double *rgBDz; /* for B*D*z */
 double *rgdTmp; /* temporary (random) vector used in different places */
 double *rgFuncValue; /* holding all fitness values (fitnessvector) */
 double *histFuncValues; /* holding historical best function values */

 size_t countevals; /* Number of function evaluations */
 size_t countinfeasible; /* Number of samples outside of domain given by bounds */
 double maxdiagC; /* max diagonal element of C */
 double mindiagC; /* min diagonal element of C */
 double maxEW; /* max Eigenwert of C */
 double minEW; /* min Eigenwert of C */
 double psL2; /* L2 norm of rgps */
 double pcL2; /* L2 norm of rgpc */

 bool flgEigensysIsUptodate;

 // Private CMA-ES-Specific Methods
 void sampleSingle(size_t sampleIdx); /* sample individual */
 void evaluateSamples(); /* evaluate all samples until done */
 void adaptC(int hsig); /* CMA-ES covariance matrix adaption */
 void updateEigensystem(int flgforce);
 void eigen(size_t N, double **C, double *diag, double **Q) const;
 size_t maxIdx(const double *rgd, size_t len) const;
 size_t minIdx(const double *rgd, size_t len) const;
 void sort_index(const double *rgFunVal, size_t *index, size_t n) const;
 bool isFeasible(size_t sampleIdx) const; /* check if sample inside lower & upper bounds */
 double doubleRangeMax(const double *rgd, size_t len) const;
 double doubleRangeMin(const double *rgd, size_t len) const;
 bool doDiagUpdate() const;
 bool isStoppingCriteriaActive(const char *criteria) const;

 // Private CCMA-ES-Specific Variables 
 size_t _numConstraints; /* number of constraints */
 double _targetSucRate; /* target success rate */
 double _beta; /* cov adaption size */
 double _cv; /* learning rate in normal vector  update */
 double _cp; /* update rate global success estimate */
 
 //TODO: check all initialization of arrays (DW)
 double globalSucRate; /* estim. global success rate */ 
 double fviability; /* viability func value */
 double frgxmean; /* function evaluation at mean */
 double frgxold; /* function evaluation prev. mean */
 size_t resampled; /* number of resampled parameters due constraint violation */
 size_t adaptionsVia; /* number of cov matrix adaptions in VIA */
 size_t adaptionsVie; /* number of cov matrix adaptions in VIE */
 size_t countcevals; /* Number of constraint evaluations */
 double *sucRates; /* constraint success rates */
 double *viabilityBounds; /* viability boundaries */
 double *maxConstraintViolations; /* max violations for VIA */ //TODO: check, same as above (DW)
 bool *viabilityImprovement; /* sample evaluations larger than fviability */ //TODO: check, not needed (DW)
 size_t *numviolations; /* number of constraint violations for each sample */
 bool **viabilityIndicator; /* constraint evaluation better than viability bound */
 double **constraintEvaluations; /* evaluation of each constraint for each sample  */
 double **v; /* normal approximation of constraints */

 // Private CCMA-ES-Specific Methods
 void setConstraints();
 void updateConstraintEvaluations();
 void updateViabilityBoundaries(/*const fp* functions, T *theta*/);
 void handleViabilityConstraints(/*fp *functions,T* theta,T **boundaries,int &resampled,int &cevals,int adapts*/);
 void handleConstraintsVie(/*fp *functions,T* theta,T **boundaries,int &resampled,int &cevals,int adapts*/);
 void updateSigmaVIE();

 // Print Methods
 void printGeneration() const;
 void printFinal() const;
};

} // namespace Korali

#endif // _KORALI_CCMAES_H_
