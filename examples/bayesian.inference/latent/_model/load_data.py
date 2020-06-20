import numpy as np


class PointsInfoStruct:

  def __init__(self,
               points,
               nPoints,
               nDimensions,
               nClusters=None,
               assignments=None):
    self.points = np.array(points)
    self.nPoints = nPoints
    self.nDimensions = nDimensions
    self.nClusters = nClusters
    self.assignments = np.array(assignments)

  def reset_points(self, new_points, new_assignments=None, new_nClusters=1):
    assert len(new_points) == len(new_assignments)
    self.nPoints = len(new_points)
    self.points = np.array(new_points)
    self.assignments = [] if new_assignments is None else new_assignments
    self.assignments = np.array(self.assignments)
    for pt in new_points:
      assert len(pt) == self.nDimensions
    self.nClusters = new_nClusters


def extr(line, toint=True, tofloat=False):
  result = line.strip("\n ").split(" ")
  if tofloat:
    result = [float(r) for r in result]
  elif toint:
    result = [int(r) for r in result]
  return result


class UnivariateData(PointsInfoStruct):

  def __init__(self):

    self.assignments = []
    self.points = []
    self.nClusters = 1

    print("Loading data from data_single.in ... \n")
    with open("_model/data_single.in", "r") as fd:
      self.nPoints, self.nDimensions = extr(fd.readline())
      for i in range(self.nPoints):
        point = extr(fd.readline(), tofloat=True)
        self.points.append(point)

    self.points = np.array(self.points)


class MultivariateData(PointsInfoStruct):

  def __init__(self):

    self.assignments = []
    self.points = []

    with open("_model/data_multiple.in", "r") as fd:
      self.nPoints, self.nDimensions, self.nClusters = extr(fd.readline())
      for i in range(self.nPoints):
        linecontents = extr(fd.readline(), toint=False)
        assert len(linecontents) == self.nDimensions + 1
        self.points.append(list(map(float, linecontents[:-1])))
        self.assignments.append(int(linecontents[-1]))

    self.points = np.array(self.points)
    self.assignments = np.array(self.assignments)
