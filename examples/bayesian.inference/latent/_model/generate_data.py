import numpy as np


def draw_from_single_gaussian(nr_points=10, dimensions=2, mu=[0, 0], sigma=1):
  ''' '''
  results, _ = draw_from_multiple_gaussian(
      nr_points, dimensions=dimensions, mus=[mu], sigma=sigma)
  return results


def draw_from_multiple_gaussian(nr_points=10,
                                dimensions=2,
                                mus=[[0, 0], [2, 2]],
                                sigma=1):
  results = np.zeros((nr_points, dimensions))
  cluster_assignment = np.zeros((nr_points,), dtype=int)
  nr_clusters = len(mus)
  for i in range(nr_points):
    c = np.random.choice(list(range(nr_clusters)))
    mu = mus[c]
    pt = np.random.normal(loc=mu, scale=sigma)
    cluster_assignment[i] = c
    results[i, :] = pt
  return results, cluster_assignment


def generate_data():
  nr_points = 10
  dimensions = 2
  single_output_file = "data_single.in"
  multiple_output_file = "data_multiple.in"

  # one of the two tutorials for SAEM assume that all mus lie in [-20, 20]**nDimensions, so try to stick with that.
  mus = [[0, 0], [2, 2]]  # [[0], [2]]
  sdev = 0.5
  nr_clusters = len(mus)

  single_data = draw_from_single_gaussian(
      nr_points, dimensions, mu=mus[1], sigma=sdev)
  multiple_data, ass = draw_from_multiple_gaussian(
      nr_points, dimensions, mus=mus, sigma=sdev)

  # store to files:
  with open(single_output_file, "w") as fd:
    d = single_data
    fd.write(str(nr_points) + " " + str(dimensions) + "\n")
    lines = [
        " ".join([str(d[i, j])
                  for j in range(dimensions)]) + "\n"
        for i in range(nr_points)
    ]
    fd.writelines(lines)
    fd.write("points (dim=" + str(dimensions) + ") ")

  # store to files:
  with open(multiple_output_file, "w") as fd:
    d = multiple_data
    fd.write(
        str(nr_points) + " " + str(dimensions) + " " + str(nr_clusters) + "\n")
    lines = [
        " ".join([str(d[i, j])
                  for j in range(dimensions)]) + " " + str(ass[i]) + "\n"
        for i in range(nr_points)
    ]
    fd.writelines(lines)
    fd.write("points (dim=" + str(dimensions) +
             ") | last column: cluster assignment")

  print("Done")


if __name__ == '__main__':
  generate_data()
