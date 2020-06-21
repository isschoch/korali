import korali

e = korali.Experiment()
k = korali.Engine()

# Example: Configuring settings 1 and 2, of string and real types, respectively.
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Setting 1"] = "String"
e["Problem"]["Setting 2"] = 1.0

# Example: Configuring setting 3, whcih contains sub-proberties to be defined.
e["Problem"]["Setting 3"]["Sub-Type"] = "myType"
e["Problem"]["Setting 3"]["Parameter 1"] = 0.0
e["Problem"]["Setting 3"]["Parameter 2"] = 1.0

# k["Solver"]["Type"] = "Optimizer/CMAES"

# k["Solver"]["Population Size"] = 32
# k["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 0.0001
# k["Solver"]["Termination Criteria"]["Max Generations"] = 1000

# Example: Defining two variables for my problem.
e["Variables"][0]["Name"] = "Thermal Conductivity"
e["Variables"][0]["Lower Bound"] = 0.0
e["Variables"][0]["Upper Bound"] = 1.0

e["Variables"][1]["Name"] = "Heat Source Position"
e["Variables"][1]["Lower Bound"] = -10.0
e["Variables"][1]["Upper Bound"] = +10.0

# defining a model function for my experiment that return F(x) and quantities of interest
def myModel(k):
    thermalConductivity = k["Parameters"][0]
    heatSourncePosition = k["Paramters"][1]
    distanceFromSource = 1.0 - heatSourncePosition
    k["Distance From Source"] = distanceFromSource
    k["F(x)"] = thermalConductivity * distanceFromSource * distanceFromSource

# e["Problem"]["Type"] = "Optimization/Stochastic"

e["Problem"]["Type"] = "Optimization/Stochastic"
e["Problem"]["Objective Function"] = myModel

# Example: Defining two variables for my problem.
e["Distributions"][0]["Name"] = "My Distribution 1"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
# e["Distributions"][0]["Minimum"] = -10.0
# e["Distributions"][0]["Maximum"] = +10.0

e["Distributions"][1]["Name"] = "My Distribution 2"
e["Distributions"][1]["Type"] = "Univariate/Normal"
e["Distributions"][1]["Mean"] = 0.0
e["Distributions"][1]["Standard Deviation"] = 5.0

e["Variables"][0]["Name"] = "Thermal Conductivity"
e["Variables"][0]["Prior Distribution"] = "My Distribution 1"

e["Variables"][1]["Name"] = "Heat Source Position"
e["Variables"][1]["Prior Distribution"] = "My Distribution 1"

# Defining conditional prio distributions for a hierarchical Bayesian problem
e["Variables"][0]["Name"] = "Psi 1"
e["Variables"][1]["Name"] = "Psi 2"

e["Distributions"][0]["Name"] = "Conditional 0"
e["Distributions"][0]["Type"] = "Univariate/Normal"
e["Distributions"][0]["Mean"] = "Psi 1"
e["Distributions"][0]["Standard Deviation"] = "Psi 2"

e["Problem"]["Conditional Priors"] = [ "Conditional 0" ]

k = korali.Engine()
k.run(e)

bestSample = e["Results"]["Best Sample"]
print('Found best sample at:')
print('Thermal Conductivity = ' + str(bestSample["Parameters"][0]))
print('Heat Source Position = ' + str(bestSample["Parameters"][1]))
print('Evaluation: ' + bestSample["F(x)"])

e["File Output"]["Path"] = "./myResultsFolder"
