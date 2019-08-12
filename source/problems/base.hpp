#ifndef _KORALI_PROBLEM_BASE_HPP_
#define _KORALI_PROBLEM_BASE_HPP_

#include "auxiliars/json.hpp"
#include "variable/variable.hpp"

namespace Korali { namespace Problem {

class Base
{
  public:

  virtual void runModel(std::vector<double>, size_t sampleId) = 0;

  virtual size_t getVariableCount() = 0;
  virtual Korali::Variable* getVariable(size_t variableId) = 0;
  virtual bool isSampleFeasible(double* sample) = 0;

  virtual double evaluateSampleFitness() = 0;
  virtual double evaluateSampleLogPrior(double* sample) = 0;

  virtual void initialize() = 0;
  virtual void finalize() = 0;

  // Serialization Methods
  virtual void getConfiguration(nlohmann::json& js) = 0;
  virtual void setConfiguration(nlohmann::json& js) = 0;
};

} } // namespace Korali::Problem


#endif // _KORALI_PROBLEM_BASE_HPP_
