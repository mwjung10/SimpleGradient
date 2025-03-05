import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy as anp


def gradient_descent(f, start, alpha, max_iter=1000, tol=1e-5):
    trajectory = [start]
    function_values = []

    x = anp.array(start, dtype=float)
    f_grad = grad(lambda x: f(x[0], x[1]))

    for _ in range(max_iter):
        grad_val = anp.array(f_grad(x))
        function_values.append(f(x[0], x[1]))

        if anp.linalg.norm(grad_val) < tol:
            break

        x = x - alpha * grad_val
        trajectory.append(x.copy())

    return anp.array(trajectory), function_values


def visualize_fun(obj_fun: callable, trajectory: np.ndarray):
    min_x, min_y = trajectory[-1]
    MIN_X = 10
    MAX_X = 10
    PLOT_STEP = 100

    x1 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    x2 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    X1, X2 = np.meshgrid(x1, x2)
    Z = obj_fun(X1, X2)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X1, X2, Z, cmap='viridis', shading='auto')
    plt.colorbar(label='Objective Function Value')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Objective Function Visualization')

    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red',
             label='Gradient Descent Steps', alpha=0.5)

    plt.scatter(min_x, min_y, color='yellow',
                label='Minimum found by gradient descent alg.', zorder=3)

    plt.legend()
    plt.show()


def function1(x1, x2):
    """Funkcja celu f(x1, x2) = x1^2 + x2^2"""
    return x1**2 + x2**2


def matyas_function(x1, x2):
    """Funkcja Matyasa"""
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def plot_alpha_effect():
    alphas = [0.01, 0.1, 0.5, 1.0]
    start = (-8, 9)

    plt.figure(figsize=(8, 6))
    for alpha in alphas:
        traj, function_values = gradient_descent(function1, start, alpha)
        plt.plot(function_values, label=f'α={alpha}')

    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Impact of Learning Rate on Gradient Descent Convergence')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_alpha_effect()

    start_points = [(-8, 9), (5, -3), (-2, -6)]
    alpha = 0.1

    for start in start_points:
        traj, function_values = gradient_descent(function1, start, alpha)
        visualize_fun(function1, traj)
