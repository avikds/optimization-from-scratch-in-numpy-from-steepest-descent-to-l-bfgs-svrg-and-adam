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

# Step 3 - LogisticProblem
from sklearn.datasets import load_breast_cancer


def load_cancer():
    # Load the breast-cancer dataset.
    data = load_breast_cancer()

    X = data.data.astype(float)
    y = data.target.astype(float)

    # Standardize every feature using population standard deviation (ddof=0).
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    X = (X - mean) / std

    # Add a leading column of ones for the intercept.
    X = np.column_stack((np.ones(X.shape[0]), X))

    return X, y


def sigmoid(z):
    # Numerically stable sigmoid implementation.
    return np.exp(-np.logaddexp(0.0, -z))


def make_logistic(N, D, seed=0):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Generate synthetic features with decreasing feature scales.
    X = rng.standard_normal((N, D)) * np.logspace(0.0, -1.0, D)

    # Generate the true weight vector.
    w_true = 2.0 * rng.standard_normal(D)

    # Generate binary labels according to the logistic probabilities.
    y = (rng.random(N) < sigmoid(X @ w_true)).astype(float)

    return X, y


class LogisticProblem:
    def __init__(self, X, y, lam):
        # Store the dataset and regularization parameter.
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.lam = float(lam)

        # Dataset dimensions.
        self.N, self.D = self.X.shape

        # Initialize computation tracking.
        self.touched = 0
        self.ledger = []

        # Cache the last parameter vector that performed a full sweep.
        self._last_swept_w = None

    @property
    def passes(self):
        # Number of equivalent passes over the dataset.
        return self.touched / self.N

    def value(self, w):
        # Calculate the regularized logistic regression objective.
        w = np.asarray(w, dtype=float)

        # Convert binary labels {0, 1} into signs {-1, +1}.
        s = 2.0 * self.y - 1.0

        # Numerically stable logistic loss.
        scores = self.X @ w
        data_loss = np.mean(np.logaddexp(0.0, -s * scores))

        # L2 regularization.
        reg_loss = 0.5 * self.lam * np.dot(w, w)

        return float(data_loss + reg_loss)

    def _charge_sweep(self, w):
        # Check whether this parameter vector has already been swept.
        w = np.asarray(w, dtype=float)

        if (
            self._last_swept_w is None
            or not np.array_equal(w, self._last_swept_w)
        ):
            # Remember a copy of the current parameter vector.
            self._last_swept_w = w.copy()

            # Charge one full pass over the dataset.
            self.touched += self.N

            # Record the current objective value in the ledger.
            self.ledger.append((self.passes, self.value(w)))

    def loss(self, w):
        # Charge a full data sweep if necessary.
        self._charge_sweep(w)

        # Return the objective value.
        return self.value(w)

    def grad(self, w):
        # Convert the parameter vector to a NumPy array.
        w = np.asarray(w, dtype=float)

        # Charge a full data sweep if necessary.
        self._charge_sweep(w)

        # Calculate predicted probabilities.
        p = sigmoid(self.X @ w)

        # Full gradient of the regularized logistic objective.
        gradient = (self.X.T @ (p - self.y)) / self.N + self.lam * w

        return gradient

    def grad_i(self, w, idx):
        # Convert parameters and indices to NumPy arrays.
        w = np.asarray(w, dtype=float)
        idx = np.atleast_1d(idx).reshape(-1)

        if idx.size == 0:
            raise ValueError("idx must contain at least one example")

        # Select the requested examples.
        X_batch = self.X[idx]
        y_batch = self.y[idx]

        # Charge only the selected examples.
        self.touched += len(idx)

        # Calculate probabilities for the selected examples.
        p = sigmoid(X_batch @ w)

        # Mean gradient over the selected examples, plus L2 regularization.
        gradient = (
            X_batch.T @ (p - y_batch)
        ) / len(idx) + self.lam * w

        return gradient

    def hess(self, w):
        # Convert the parameter vector to a NumPy array.
        w = np.asarray(w, dtype=float)

        # Charge N * D examples for computing the full Hessian.
        self.touched += self.N * self.D

        # Calculate predicted probabilities.
        p = sigmoid(self.X @ w)

        # Calculate the diagonal weights p * (1 - p).
        weights = p * (1.0 - p)

        # Construct the weighted feature matrix.
        weighted_X = self.X * weights[:, np.newaxis]

        # Hessian of the regularized logistic objective.
        hessian = (
            self.X.T @ weighted_X
        ) / self.N + self.lam * np.eye(self.D)

        return hessian

    def checkpoint(self, w):
        # Record the current passes and objective value without charging.
        self.ledger.append((self.passes, self.value(w)))

