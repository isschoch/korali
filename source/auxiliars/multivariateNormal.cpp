#include "auxiliars/multivariateNormal.hpp"
#include <auxiliars/logger.hpp>

Korali::MultivariateNormal::MultivariateNormal(size_t seed)
{
 _range = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(_range, seed);
}

Korali::MultivariateNormal::~MultivariateNormal()
{
 gsl_rng_free(_range);
}

void Korali::MultivariateNormal::getDensityVector(double *x, double* result, size_t n)
{
 if (_workVector.size() != n)
  Korali::logError("MultivariateNormal::getDensityVector Error - requested %lu densities, but distribution is configured with %lu.\n", n, _workVector.size());

 gsl_vector_view _input_view = gsl_vector_view_array(x, n);

 gsl_ran_multivariate_gaussian_pdf(&_input_view.vector, &_mean_view.vector, &_sigma_view.matrix, result, &_work_view.vector);
}

void Korali::MultivariateNormal::getLogDensityVector(double *x, double* result, size_t n)
{
 gsl_vector_view _input_view = gsl_vector_view_array(x, n);

 gsl_ran_multivariate_gaussian_log_pdf(&_input_view.vector, &_mean_view.vector, &_sigma_view.matrix, result, &_work_view.vector);
}

void Korali::MultivariateNormal::getRandomVector(double *x, size_t n)
{
 gsl_vector_view _output_view = gsl_vector_view_array(x, n);

 gsl_ran_multivariate_gaussian(_range, &_mean_view.vector, &_sigma_view.matrix, &_output_view.vector);
}

void Korali::MultivariateNormal::updateDistribution()
{
 size_t covarianceMatrixSize = _covarianceMatrix.size();

 size_t sideSize = sqrt(covarianceMatrixSize);
 if ((sideSize * sideSize) != covarianceMatrixSize)
  Korali::logError("Size of Multivariate Normal covariance matrix size (%lu) is not a perfect square number.\n", covarianceMatrixSize);

 size_t meanSize = _meanVector.size();
 if (sideSize != meanSize) Korali::logError("Size of Multivariate Normal mean vector (%lu) is not the same as the side of covariance matrix (%lu).\n", meanSize, sideSize);

 _workVector.resize(meanSize);

 _sigma_view  = gsl_matrix_view_array(_covarianceMatrix.data(), sideSize,sideSize);
 _mean_view   = gsl_vector_view_array(_meanVector.data(), meanSize);
 _work_view   = gsl_vector_view_array(_workVector.data(), meanSize);
}

void Korali::MultivariateNormal::setProperty(std::string propertyName, std::vector<double> values)
{
 bool recognizedProperty = false;
 if (propertyName == "Mean Vector") { _meanVector = values; recognizedProperty = true; }
 if (propertyName == "Covariance Matrix") { _covarianceMatrix = values; recognizedProperty = true; }
 if (recognizedProperty == false) Korali::logError("Unrecognized property: %s for the Multivariate Normal distribution", propertyName.c_str());
}
