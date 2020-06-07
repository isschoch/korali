import numpy as np
import matplotlib.pyplot as plt


def explicit_euler(q_init, p_init, grad_U, grad_K, step_size, num_steps):
    q = np.zeros(shape=(len(q_init), num_steps + 1), dtype=float)
    p = np.zeros(shape=(len(q_init), num_steps + 1), dtype=float)

    q[:, 0] = q_init
    p[:, 0] = p_init

    for i in range(num_steps):
        q[:, i+1] = q[:, i] + step_size * grad_K(p[:, i])
        p[:, i+1] = p[:, i] - step_size * grad_U(q[:, i])

    return q, p


def leapfrog(q_init, p_init, grad_U, grad_K, step_size, num_steps):
    q = np.zeros(shape=(len(q_init), num_steps + 1), dtype=float)
    p = np.zeros(shape=(len(q_init), num_steps + 1), dtype=float)
    p_midsteps = np.zeros(shape=(len(q_init), num_steps + 1), dtype=float)

    q[:, 0] = q_init
    p[:, 0] = p_init

    for i in range(num_steps):
        p_midsteps[:, i+1] = p[:, i] - step_size / 2. * grad_U(q[:, i])
        q[:, i+1] = q[:, i] + step_size * grad_K(p_midsteps[:, i+1])
        p[:, i+1] = p_midsteps[:, i+1] - step_size / 2. * grad_U(q[:, i+1])

    return q, p


def test_euler(q_init, p_init, U, grad_U, grad_K, step_size, num_steps):
    q, p = explicit_euler(q_init, p_init, grad_U, grad_K, step_size, num_steps)
    print("q = ", q)
    print("p = ", p)

    plt.xlabel("q")
    plt.ylabel("p")
    plt.title("Explicit Euler")
    plt.grid(":")
    plt.plot(q[0, :], p[0, :], 'r-')
    plt.savefig("explicit_euler_test.png")
    plt.close()


def test_leapfrog(q_init, p_init, U, grad_U, grad_K, step_size, num_steps):
    q, p = leapfrog(q_init, p_init, grad_U, grad_K, step_size, num_steps)
    print("q = ", q)
    print("p = ", p)

    plt.xlabel("q")
    plt.ylabel("p")
    plt.title("Leapfrog")
    plt.grid(":")
    plt.plot(q[0, :], p[0, :], 'r-')
    plt.savefig("leapfrog_test.png")
    plt.close()


def test():
    q_init = np.array([0.])
    p_init = np.array([1.])
    mass = np.ones(shape=(len(p_init)), dtype=float)
    mass_matrix = np.diag(mass)

    def U(q): return q ** 2 / 2

    def grad_U(q): return q

    def K(p): return p ** 2 / 2

    def grad_K(p): return np.matmul(np.linalg.inv(mass_matrix), p)

    # values from Figure 1 in paper "MCMC using Hamiltonian dynamics"
    step_size = 0.3
    num_steps = 20

    test_euler(q_init, p_init, U, grad_U, grad_K, step_size, num_steps)
    test_leapfrog(q_init, p_init, U, grad_U, grad_K, step_size, num_steps)
