"""
Optimization from Scratch in NumPy: From Steepest Descent to L-BFGS, SVRG and Adam

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_quadratic
import numpy as np

A_SKEWED = np.array([[20.0, 5.0], [5.0, 2.0]])
A_ROUND = np.array([[20.0, 5.0], [5.0, 16.0]])
B_FIG = np.array([-14.0, -6.0])


def condition_number(A):
    # Calculate the eigenvalues of the symmetric matrix.
    eigenvalues = np.linalg.eigvalsh(A)

    # Condition number = largest eigenvalue / smallest eigenvalue.
    return float(eigenvalues[-1] / eigenvalues[0])


def make_quadratic(kappa, dim, seed=0):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Generate a random orthogonal matrix using QR decomposition.
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))

    # Generate logarithmically spaced eigenvalues from 1 to kappa.
    eigenvalues = np.logspace(0.0, np.log10(kappa), dim)

    # Construct the positive-definite matrix A.
    A = Q @ np.diag(eigenvalues) @ Q.T

    # Ensure that A is exactly symmetric up to floating-point precision.
    A = 0.5 * (A + A.T)

    # Generate the linear term b.
    b = rng.standard_normal(dim)

    return A, b


def quadratic_loss(theta, A, b, c=0.0):
    # Convert theta to a NumPy array to support list and array inputs.
    theta = np.asarray(theta)

    # Calculate the quadratic objective:
    # L(theta) = 0.5 * theta.T @ A @ theta + b.T @ theta + c
    loss = 0.5 * theta.T @ A @ theta + b.T @ theta + c

    return float(loss)


def quadratic_grad(theta, A, b):
    # Convert theta to a NumPy array.
    theta = np.asarray(theta)

    # Gradient of the quadratic objective: A @ theta + b
    return A @ theta + b

# Step 2 - valley_loss
def valley_loss(theta):
    # Convert theta to a NumPy array.
    theta = np.asarray(theta, dtype=float)

    x, y = theta

    # Calculate the curved valley objective function.
    loss = 0.5 * (x**2 - y)**2 + 0.5 * (x - 1)**2

    return float(loss)


def valley_grad(theta):
    # Convert theta to a NumPy array.
    theta = np.asarray(theta, dtype=float)

    x, y = theta

    # Calculate the gradient analytically.
    grad = np.array([
        2.0 * x * (x**2 - y) + (x - 1.0),
        y - x**2
    ], dtype=float)

    return grad


def valley_hessian(theta):
    # Convert theta to a NumPy array.
    theta = np.asarray(theta, dtype=float)

    x, y = theta

    # Calculate the Hessian matrix analytically.
    hessian = np.array([
        [6.0 * x**2 - 2.0 * y + 1.0, -2.0 * x],
        [-2.0 * x, 1.0]
    ], dtype=float)

    return hessian

