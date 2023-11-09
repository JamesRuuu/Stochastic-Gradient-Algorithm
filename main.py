import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    return (10 * pow(x[0], 2) + pow(x[1], 2)) / 2


def grad_f(x):
    return np.array([20 * x[0] / 2, 2 * x[1] / 2])


def stochastic_gradient_descent(lr, pc):
    x = np.random.randn(2)  # randomly initialize x 返回的结果服从标准正态分布, ndarray数组
    list_x = []  # 存储每一次梯度下降时的x1, x2
    while True:
        gradient = grad_f(x)
        x_new = x - lr * gradient
        if np.abs(f(x_new) - f(x)) < pc:
            break
        x = x_new
        list_x.append(x)
    return x_new, list_x


if __name__ == '__main__':
    learning_rate = 0.05  # learning rate
    precision = 1e-5  # computation precision
    result, coordinates = stochastic_gradient_descent(learning_rate, precision)
    print("Final Point (x1, x2):", result)
    print("Minimum Value:", f(result))

    x1_coords = [x[0] for x in coordinates]
    x2_coords = [x[1] for x in coordinates]
    fx_coords = [f(x) for x in coordinates]

    # Plot the series of points
    plt.plot(x1_coords, x2_coords, '-o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Stochastic Gradient Descent')
    plt.grid(True)
    plt.show()

    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
    z_mesh = f([x_mesh, y_mesh])

    # Plot the surface of the function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.set_title('Surface plot of f(x)')

    # Plot the series of points on the surface
    ax.plot(x1_coords, x2_coords, fx_coords, 'ro-', label='Points on f(x)')
    ax.legend()

    # Show the plot
    plt.show()