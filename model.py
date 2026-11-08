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

# Step 4 - gd_fixed
def gd_fixed(grad, theta0, eta, n_iter):
    # Create a floating-point copy of the initial parameters.
    # This ensures theta0 is not modified.
    theta = np.array(theta0, dtype=float, copy=True)

    # Store the initial point and every subsequent iterate.
    path = np.empty((n_iter + 1, theta.size), dtype=float)
    path[0] = theta

    # Perform fixed-step gradient descent.
    for t in range(n_iter):
        theta = theta - eta * np.asarray(grad(theta), dtype=float)

        # Store the current iterate.
        path[t + 1] = theta

    return path


def max_stable_step(A):
    # Calculate the eigenvalues of the symmetric matrix.
    eigenvalues = np.linalg.eigvalsh(A)

    # Maximum stable step size: 2 / largest eigenvalue.
    return float(2.0 / eigenvalues[-1])

# Step 5 - steepest_descent_exact
def exact_step(A, b, theta, d):
    # Convert inputs to floating-point NumPy arrays.
    theta = np.asarray(theta, dtype=float)
    d = np.asarray(d, dtype=float)

    # Gradient of the quadratic objective at theta.
    grad = A @ theta + b

    # Exact line search:
    # eta = -(d.T @ grad) / (d.T @ A @ d)
    numerator = -np.dot(d, grad)
    denominator = np.dot(d, A @ d)

    if denominator == 0.0:
        return 0.0

    return float(numerator / denominator)


def steepest_descent_exact(A, b, theta0, n_iter):
    # Create a floating-point copy of the initial point.
    theta = np.array(theta0, dtype=float, copy=True)

    # Store every iterate, beginning with the initial point.
    path = [theta.copy()]

    for _ in range(n_iter):
        # Calculate the gradient.
        grad = A @ theta + b

        # Steepest descent direction.
        d = -grad

        # Stop before taking a step if the gradient norm is sufficiently small.
        if np.dot(d, d) < 1e-28:
            break

        # Calculate the exact optimal step size.
        eta = exact_step(A, b, theta, d)

        # Update the parameters.
        theta = theta + eta * d

        # Store the new iterate.
        path.append(theta.copy())

    return np.array(path, dtype=float)


def rate_bound(kappa):
    # Theoretical convergence-rate bound from equation 8.29.
    mu = ((kappa - 1.0) / (kappa + 1.0)) ** 2

    return float(mu)


def worst_case_start(A, b):
    # Calculate the eigenvalues and corresponding eigenvectors.
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Smallest and largest eigenvalues.
    lambda_min = eigenvalues[0]
    lambda_max = eigenvalues[-1]

    # Corresponding normalized eigenvectors.
    q_min = eigenvectors[:, 0]
    q_max = eigenvectors[:, -1]

    # Calculate the optimal solution of the quadratic objective.
    theta_star = np.linalg.solve(A, -b)

    # Construct the worst-case starting point.
    theta0 = (
        theta_star
        + q_min / lambda_min
        + q_max / lambda_max
    )

    return theta0


def observed_rates(losses, loss_star):
    # Convert the loss sequence to a floating-point NumPy array.
    losses = np.asarray(losses, dtype=float)

    # Calculate consecutive ratios of objective gaps.
    rates = (
        (losses[1:] - loss_star)
        / (losses[:-1] - loss_star)
    )

    return rates

# Step 6 - steepest_descent_armijo
def armijo_backtrack(f, theta, d, g, eta0=1.0, c=1e-4, rho=0.5, max_halvings=60):
    # Convert inputs to floating-point NumPy arrays.
    theta = np.asarray(theta, dtype=float)
    d = np.asarray(d, dtype=float)
    g = np.asarray(g, dtype=float)

    # Evaluate the objective at the current point.
    f_theta = f(theta)

    # Calculate the directional derivative.
    directional_derivative = np.dot(g, d)

    # Start with the initial trial step size.
    eta = float(eta0)

    # Try at most max_halvings candidate step sizes.
    for _ in range(max_halvings):
        # Evaluate the objective at the trial point.
        f_trial = f(theta + eta * d)

        # Check the Armijo sufficient-decrease condition.
        if f_trial <= f_theta + c * eta * directional_derivative:
            return float(eta)

        # Reduce the step size after a failed trial.
        eta *= rho

    # Return the step size after the final reduction if no trial passed.
    return float(eta)


def steepest_descent_armijo(f, grad, theta0, n_iter, eta0=1.0, c=1e-4, rho=0.5, warm_start=False):
    # Create a floating-point copy of the initial parameters.
    # This ensures theta0 is not modified.
    theta = np.array(theta0, dtype=float, copy=True)

    # Store every iterate, starting with theta0.
    path = np.empty((n_iter + 1, theta.size), dtype=float)
    path[0] = theta

    # Store the previously accepted step size for warm starting.
    previous_eta = None

    # Perform steepest descent iterations.
    for t in range(n_iter):
        # Calculate the gradient at the current point.
        g = np.asarray(grad(theta), dtype=float)

        # Steepest descent direction.
        d = -g

        # Determine the initial trial step size.
        if warm_start and t > 0:
            # Start from twice the previous accepted step size.
            trial_eta0 = 2.0 * previous_eta
        else:
            # Use the original initial step size.
            trial_eta0 = eta0

        # Find the step size using Armijo backtracking.
        eta = armijo_backtrack(
            f,
            theta,
            d,
            g,
            eta0=trial_eta0,
            c=c,
            rho=rho
        )

        # Update the parameters.
        theta = theta + eta * d

        # Remember the accepted step size.
        previous_eta = eta

        # Store the current iterate.
        path[t + 1] = theta

    return path

# Step 7 - heavy_ball
def heavy_ball(grad, theta0, eta, beta, n_iter):
    # Create a floating-point copy of the initial parameters.
    theta = np.array(theta0, dtype=float, copy=True)

    # Initialize the momentum vector to zero.
    momentum = np.zeros_like(theta)

    # Store all iterates, including the initial point.
    path = np.empty((n_iter + 1, theta.size), dtype=float)
    path[0] = theta

    for t in range(n_iter):
        # Update the momentum using the current gradient.
        momentum = beta * momentum + np.asarray(grad(theta), dtype=float)

        # Update the parameters using the momentum.
        theta = theta - eta * momentum

        # Store the current iterate.
        path[t + 1] = theta

    return path


def nesterov(grad, theta0, eta, beta, n_iter):
    # Create a floating-point copy of the initial parameters.
    theta = np.array(theta0, dtype=float, copy=True)

    # Initialize the previous iterate as theta_(-1) = theta_0.
    theta_prev = theta.copy()

    # Store all iterates, including the initial point.
    path = np.empty((n_iter + 1, theta.size), dtype=float)
    path[0] = theta

    for t in range(n_iter):
        # Calculate the look-ahead position.
        theta_lookahead = theta + beta * (theta - theta_prev)

        # Evaluate the gradient at the look-ahead position.
        g = np.asarray(grad(theta_lookahead), dtype=float)

        # Perform the gradient update.
        theta_next = theta_lookahead - eta * g

        # Update the previous and current iterates.
        theta_prev = theta.copy()
        theta = theta_next

        # Store the current iterate.
        path[t + 1] = theta

    return path


def tuned_momentum(m, L):
    # Calculate the square roots of the strong convexity
    # parameter and the smoothness constant.
    sqrt_m = np.sqrt(m)
    sqrt_L = np.sqrt(L)

    # Heavy-ball optimal parameters.
    eta_hb = 4.0 / (sqrt_L + sqrt_m)**2
    beta_hb = ((sqrt_L - sqrt_m) / (sqrt_L + sqrt_m))**2

    # Nesterov optimal parameters.
    eta_nag = 1.0 / L
    beta_nag = (sqrt_L - sqrt_m) / (sqrt_L + sqrt_m)

    # Return the parameters as Python floats.
    return {
        "heavy_ball": (float(eta_hb), float(beta_hb)),
        "nesterov": (float(eta_nag), float(beta_nag))
    }


def iters_to_gap(path, f, f_star, tol):
    # Convert the iteration path to a NumPy array.
    path = np.asarray(path, dtype=float)

    # Calculate the initial objective value.
    initial_loss = float(f(path[0]))

    # Calculate the target relative objective gap.
    target_gap = tol * (initial_loss - f_star)

    # Check every iterate, including the initial point.
    for t, theta in enumerate(path):
        # Evaluate the objective at the current iterate.
        loss = float(f(theta))

        # NaN losses do not count as reaching the target.
        if np.isnan(loss):
            continue

        # Check whether the required objective gap is reached.
        if loss - f_star <= target_gap:
            return int(t)

    # Return None if no iterate reaches the target.
    return None

# Step 8 - newton_minimize
def newton_minimize(f, grad, hess, theta0, n_iter, line_search=True, tol=1e-10):
    # Create a floating-point copy of the initial parameters.
    # This ensures theta0 is not modified.
    theta = np.array(theta0, dtype=float, copy=True)

    # Store all iterates, starting with the initial point.
    path = [theta.copy()]

    for _ in range(n_iter):
        # Calculate the gradient at the current point.
        g = np.asarray(grad(theta), dtype=float)

        # Stop if the gradient norm is sufficiently small.
        if np.linalg.norm(g) <= tol:
            break

        # Calculate the Hessian at the current point.
        H = np.asarray(hess(theta), dtype=float)

        # Solve the Newton system:
        # H @ d = -g
        d = np.linalg.solve(H, -g)

        # Choose the step size.
        if line_search:
            # Use Armijo backtracking with an initial step size of 1.0.
            eta = armijo_backtrack(
                f,
                theta,
                d,
                g,
                eta0=1.0
            )
        else:
            # Use the full Newton step without line search.
            eta = 1.0

        # Update the parameters.
        theta = theta + eta * d

        # Store the current iterate.
        path.append(theta.copy())

    return np.array(path, dtype=float)

# Step 9 - trust_region_minimize
def damped_newton_step(g, H, lam):
    # Convert inputs to floating-point NumPy arrays.
    g = np.asarray(g, dtype=float)
    H = np.asarray(H, dtype=float)

    # Construct the damped Hessian: H + lambda * I.
    H_damped = H + lam * np.eye(H.shape[0])

    # Solve the damped Newton system:
    # (H + lambda * I) @ delta = -g
    delta = np.linalg.solve(H_damped, -g)

    return delta


def trust_region_minimize(f, grad, hess, theta0, n_iter=200, lam0=1.0, tol=1e-8):
    # Create a floating-point copy of the initial parameters.
    theta = np.array(theta0, dtype=float, copy=True)

    # Initialize the damping parameter.
    lam = float(lam0)

    # Store accepted iterates, beginning with the initial point.
    path = [theta.copy()]

    for _ in range(n_iter):
        # Step 1: Compute the gradient and Hessian.
        g = np.asarray(grad(theta), dtype=float)
        H = np.asarray(hess(theta), dtype=float)

        # Stop if the gradient norm is sufficiently small.
        if np.linalg.norm(g) <= tol:
            break

        # Step 2: Adjust lambda to ensure the damped Hessian
        # is positive definite.
        smallest_eigenvalue = np.linalg.eigvalsh(H)[0]

        lam = max(lam, -1.001 * smallest_eigenvalue)

        # Step 3: Compute the damped Newton direction.
        delta = damped_newton_step(g, H, lam)

        # Calculate the predicted reduction from the quadratic model.
        pred = -(
            np.dot(g, delta)
            + 0.5 * np.dot(delta, H @ delta)
        )

        # Step 4: If predicted reduction is extremely small,
        # take the step, record it, and stop.
        if pred < 1e-15:
            theta = theta + delta
            path.append(theta.copy())
            break

        # Step 5: Calculate the actual-to-predicted reduction ratio.
        actual = f(theta) - f(theta + delta)
        rho = actual / pred

        # Step 6: Adjust the damping parameter according to rho.
        if rho > 0.75:
            lam *= 0.5
        elif rho < 0.25:
            lam *= 4.0

        # Step 7: Accept the step only when actual reduction is positive.
        if rho > 0.0:
            theta = theta + delta
            path.append(theta.copy())

        # Otherwise, reject the step and remain at the current point.

    return np.array(path, dtype=float)

# Step 10 - bfgs_update_inverse
def bfgs_update_hessian(B, s, y):
    # Convert inputs to floating-point NumPy arrays.
    B = np.asarray(B, dtype=float)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)

    # Calculate the necessary matrix and vector products.
    Bs = B @ s
    ys = np.dot(y, s)
    sBs = np.dot(s, Bs)

    # BFGS Hessian update:
    # B_new = B + (y @ y.T) / (y.T @ s)
    #           - (B @ s @ s.T @ B) / (s.T @ B @ s)
    B_new = (
        B
        + np.outer(y, y) / ys
        - np.outer(Bs, Bs) / sBs
    )

    return B_new


def bfgs_update_inverse(C, s, y):
    # Convert inputs to floating-point NumPy arrays.
    C = np.asarray(C, dtype=float)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)

    # Calculate the curvature term.
    ys = np.dot(y, s)

    # Identity matrix with the same dimension as C.
    I = np.eye(C.shape[0])

    # Construct the two transformation matrices.
    left = I - np.outer(s, y) / ys
    right = I - np.outer(y, s) / ys

    # BFGS inverse Hessian update.
    C_new = (
        left @ C @ right
        + np.outer(s, s) / ys
    )

    return C_new

# Step 11 - wolfe_line_search
def wolfe_line_search(f, grad, theta, d, eta0=1.0, c1=1e-4, c2=0.9, max_iter=40):
    # Convert inputs to floating-point NumPy arrays.
    theta = np.asarray(theta, dtype=float)
    d = np.asarray(d, dtype=float)

    # Evaluate the objective and gradient at the initial point.
    f0 = float(f(theta))
    g0 = np.asarray(grad(theta), dtype=float)

    # Initial directional derivative.
    dg0 = np.dot(d, g0)

    # Initialize the step-size bracket.
    lo = 0.0
    hi = np.inf

    # Initial trial step size.
    eta = float(eta0)

    for _ in range(max_iter):
        # Evaluate the objective at the trial point.
        theta_trial = theta + eta * d
        f_trial = float(f(theta_trial))

        # Check the Armijo sufficient-decrease condition.
        armijo_rhs = f0 + c1 * eta * dg0

        if f_trial > armijo_rhs:
            # Armijo condition failed.
            hi = eta

            # Bisect the current interval.
            eta = (lo + hi) / 2.0

        else:
            # Evaluate the gradient at the trial point.
            g_trial = np.asarray(grad(theta_trial), dtype=float)

            # Calculate the directional derivative at the trial point.
            dg_trial = np.dot(d, g_trial)

            # Check the weak Wolfe curvature condition.
            if dg_trial < c2 * dg0:
                # Curvature condition failed.
                lo = eta

                # Expand the step size if the upper bound is infinite.
                if np.isinf(hi):
                    eta = 2.0 * eta
                else:
                    # Otherwise, bisect the current interval.
                    eta = (lo + hi) / 2.0

            else:
                # Both weak Wolfe conditions are satisfied.
                return float(eta)

    # Return the current step size if the maximum iterations are reached.
    return float(eta)

# Step 12 - bfgs_minimize
def bfgs_minimize(f, grad, theta0, n_iter=200, tol=1e-8):
    # Create a floating-point copy of the initial parameters.
    theta = np.array(theta0, dtype=float, copy=True)

    # Initialize the inverse Hessian approximation to the identity matrix.
    C = np.eye(theta.size)

    # Store all iterates, starting with the initial point.
    path = [theta.copy()]

    for _ in range(n_iter):
        # Calculate the gradient at the current point.
        g = np.asarray(grad(theta), dtype=float)

        # Stop if the gradient norm is sufficiently small.
        if np.linalg.norm(g) <= tol:
            break

        # Calculate the search direction using the inverse Hessian.
        d = -C @ g

        # Find the step size using the weak Wolfe line search.
        eta = wolfe_line_search(
            f,
            grad,
            theta,
            d
        )

        # Calculate the next parameter vector.
        theta_new = theta + eta * d

        # Calculate the parameter difference.
        s = theta_new - theta

        # Calculate the gradient difference.
        y = np.asarray(grad(theta_new), dtype=float) - g

        # Update the inverse Hessian approximation only
        # when the curvature condition is sufficiently positive.
        if np.dot(y, s) > 1e-12:
            C = bfgs_update_inverse(C, s, y)

        # Accept the new parameter vector.
        theta = theta_new

        # Store the accepted iterate.
        path.append(theta.copy())

    # Return the complete iteration path and final inverse Hessian.
    return np.array(path, dtype=float), C

# Step 13 - lbfgs_direction
def lbfgs_direction(g, s_list, y_list):
    # Convert the gradient into a floating-point copy.
    # This ensures the original gradient is not modified.
    q = np.array(g, dtype=float, copy=True)

    # Store the parameter and gradient difference pairs.
    # The pairs are ordered from oldest to newest.
    pairs = list(zip(s_list, y_list))

    # Store alpha and rho values for each pair.
    alpha_values = [0.0] * len(pairs)
    rho_values = [0.0] * len(pairs)

    # First loop: process pairs from newest to oldest.
    for i in range(len(pairs) - 1, -1, -1):
        s = np.asarray(pairs[i][0], dtype=float)
        y = np.asarray(pairs[i][1], dtype=float)

        # Calculate the reciprocal curvature term.
        rho = 1.0 / np.dot(y, s)

        # Calculate alpha.
        alpha = rho * np.dot(s, q)

        # Store the values for the second loop.
        rho_values[i] = rho
        alpha_values[i] = alpha

        # Update q.
        q = q - alpha * y

    # Step 2: Calculate the initial scaling factor.
    if len(pairs) > 0:
        # Use the newest stored pair.
        s_new = np.asarray(pairs[-1][0], dtype=float)
        y_new = np.asarray(pairs[-1][1], dtype=float)

        gamma = np.dot(s_new, y_new) / np.dot(y_new, y_new)
    else:
        # Use the identity scaling when no pairs are available.
        gamma = 1.0

    # Initialize r using the scaling factor.
    r = gamma * q

    # Second loop: process pairs from oldest to newest.
    for i in range(len(pairs)):
        s = np.asarray(pairs[i][0], dtype=float)
        y = np.asarray(pairs[i][1], dtype=float)

        # Calculate beta.
        beta = rho_values[i] * np.dot(y, r)

        # Update r.
        r = r + (alpha_values[i] - beta) * s

    # Return the L-BFGS search direction.
    return -r

# Step 14 - lbfgs_minimize
def lbfgs_minimize(f, grad, theta0, memory=10, n_iter=500, tol=1e-8):
    # Create a floating-point copy of the initial parameters.
    theta = np.array(theta0, dtype=float, copy=True)

    # Initialize empty lists for storing parameter and gradient differences.
    # Pairs are stored from oldest to newest.
    s_list = []
    y_list = []

    # Store all iterates, starting with the initial point.
    path = [theta.copy()]

    for _ in range(n_iter):
        # Calculate the gradient at the current point.
        g = np.asarray(grad(theta), dtype=float)

        # Stop if the gradient norm is sufficiently small.
        if np.linalg.norm(g) <= tol:
            break

        # Calculate the L-BFGS search direction using two-loop recursion.
        d = lbfgs_direction(g, s_list, y_list)

        # Determine the step size using the weak Wolfe line search.
        eta = wolfe_line_search(
            f,
            grad,
            theta,
            d
        )

        # Calculate the next parameter vector.
        theta_new = theta + eta * d

        # Calculate the parameter difference.
        s = theta_new - theta

        # Calculate the gradient difference.
        g_new = np.asarray(grad(theta_new), dtype=float)
        y = g_new - g

        # Update the stored pairs only when the curvature condition
        # is sufficiently positive.
        if np.dot(y, s) > 1e-12:
            # Store copies of the new parameter and gradient differences.
            s_list.append(s.copy())
            y_list.append(y.copy())

            # Remove the oldest pair if memory capacity is exceeded.
            if len(s_list) > memory:
                s_list.pop(0)
                y_list.pop(0)

        # Accept the new parameter vector.
        theta = theta_new

        # Store the accepted iterate.
        path.append(theta.copy())

    # Return the complete iteration path.
    return np.array(path, dtype=float)

# Step 15 - lms
def lms_data():
    # Initialize the random number generator with the specified seed.
    rs = np.random.RandomState(0)

    # Generate 21 input values uniformly between -2 and 2.
    x = rs.uniform(-2.0, 2.0, 21)

    # Construct the design matrix with an intercept column.
    X = np.column_stack([np.ones(21), x])

    # Generate target values with additive Gaussian noise.
    y = (
        1.45 * X[:, 0]
        + 0.92 * X[:, 1]
        + rs.normal(0.0, 0.1, 21)
    )

    return X, y


def lms(X, y, theta0, eta_fn, n_updates):
    # Convert inputs to floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Create a floating-point copy of the initial parameters.
    # This ensures theta0 is not modified.
    theta = np.array(theta0, dtype=float, copy=True)

    # Obtain the number of training examples and parameters.
    N, D = X.shape

    # Store all iterates, beginning with the initial parameters.
    path = np.empty((n_updates + 1, D), dtype=float)
    path[0] = theta

    # Perform sequential LMS updates.
    for t in range(1, n_updates + 1):
        # Select the next training example in sequential order.
        n = (t - 1) % N

        # Extract the current input and target.
        x_n = X[n]
        y_n = y[n]

        # Calculate the prediction error.
        error = np.dot(theta, x_n) - y_n

        # Obtain the learning rate for the current update.
        eta = eta_fn(t)

        # Apply the LMS update.
        theta = theta - eta * error * x_n

        # Store the updated parameter vector.
        path[t] = theta

    return path

# Step 16 - lr_schedule
def lr_schedule(
    kind,
    t,
    eta0=1.0,
    gamma=0.9,
    every=10,
    lam=0.1,
    beta=1.0,
    alpha=0.5,
    t_warm=10,
    t_total=100,
    period=20,
    eta_min=0.0
):
    # Constant learning-rate schedule.
    if kind == "constant":
        eta = eta0

    # Piecewise constant schedule (step decay).
    elif kind == "piecewise":
        eta = eta0 * gamma ** (t // every)

    # Exponential decay schedule.
    elif kind == "exponential":
        eta = eta0 * np.exp(-lam * t)

    # Polynomial decay schedule.
    elif kind == "polynomial":
        eta = eta0 * (beta * t + 1) ** (-alpha)

    # Warmup followed by cosine decay.
    elif kind == "warmup_cosine":
        if t < t_warm:
            # Linear warmup phase.
            eta = eta0 * (t + 1) / t_warm
        else:
            # Cosine decay phase.
            frac = (t - t_warm) / max(1, t_total - t_warm)

            eta = eta0 * 0.5 * (
                1.0 + np.cos(np.pi * min(frac, 1.0))
            )

    # Triangular cyclic learning-rate schedule.
    elif kind == "triangular":
        phase = (t % period) / period

        eta = eta_min + (eta0 - eta_min) * (
            1.0 - abs(2.0 * phase - 1.0)
        )

    else:
        raise ValueError(f"Unknown learning-rate schedule: {kind}")

    return float(eta)


def robbins_monro_ratio(kind, T, **params):
    # Evaluate the learning-rate schedule for t = 0, ..., T-1.
    etas = np.array(
        [
            lr_schedule(kind, t, **params)
            for t in range(T)
        ],
        dtype=float
    )

    # Calculate the sum of squared learning rates.
    sum_squared = np.sum(etas ** 2)

    # Calculate the sum of learning rates.
    sum_linear = np.sum(etas)

    # Avoid division by zero.
    if sum_linear == 0.0:
        raise ValueError("The sum of learning rates must be nonzero.")

    # Return the Robbins-Monro ratio as a Python float.
    return float(sum_squared / sum_linear)

# Step 17 - sgd_minibatch
def sgd_minibatch(problem, w0, schedule, batch_size, n_epochs, seed=0, average_from=0):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Create a floating-point copy of the initial parameters.
    w = np.array(w0, dtype=float, copy=True)

    # Initialize the Polyak-Ruppert running average.
    avg = w.copy()

    # Number of iterates included in the running average.
    k = 0

    # Count the total number of minibatch updates.
    t = 0

    # Perform the specified number of epochs.
    for epoch in range(n_epochs):

        # Generate a new random permutation at the beginning of each epoch.
        perm = rng.permutation(problem.N)

        # Process the shuffled dataset in consecutive minibatches.
        # The final minibatch may contain fewer examples.
        for start in range(0, problem.N, batch_size):

            # Extract the current minibatch.
            chunk = perm[start:start + batch_size]

            # Obtain the learning rate for the current update.
            eta = schedule(t)

            # Calculate the minibatch gradient and update the parameters.
            w = w - eta * problem.grad_i(w, chunk)

            # Increment the update counter.
            t += 1

            # Update the running average after the specified threshold.
            if t > average_from:
                k += 1
                avg = avg + (w - avg) / k

        # Record the objective value and data passes after each epoch.
        problem.checkpoint(w)

    # Return the final parameters and their running average.
    return {
        "w": w,
        "avg": avg
    }

# Step 18 - svrg
def max_smoothness(problem):
    # Calculate the squared Euclidean norm of each data row.
    row_norms_squared = np.sum(problem.X ** 2, axis=1)

    # Maximum smoothness bound:
    # L_max = max_n(||x_n||^2 / 4) + lambda
    L_max = np.max(row_norms_squared) / 4.0 + problem.lam

    return float(L_max)


def svrg_gradient(problem, w, snapshot, mu, n):
    # Calculate the stochastic gradient at the current parameters.
    grad_current = problem.grad_i(w, n)

    # Calculate the stochastic gradient at the snapshot.
    grad_snapshot = problem.grad_i(snapshot, n)

    # SVRG variance-reduced gradient estimator.
    gradient = grad_current - grad_snapshot + mu

    return gradient


def svrg(problem, w0, eta, n_epochs, inner=None, seed=0):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Create a floating-point copy of the initial parameters.
    w = np.array(w0, dtype=float, copy=True)

    # Default number of inner iterations to the dataset size.
    if inner is None:
        inner = problem.N

    # Perform the specified number of epochs.
    for epoch in range(n_epochs):

        # Take a copy of the current parameters as the snapshot.
        snapshot = w.copy()

        # Calculate the full gradient at the snapshot.
        mu = problem.grad(snapshot)

        # Perform the stochastic inner updates.
        for _ in range(inner):

            # Randomly select one training example.
            n = int(rng.integers(problem.N))

            # Calculate the variance-reduced gradient.
            v = svrg_gradient(
                problem,
                w,
                snapshot,
                mu,
                n
            )

            # Update the parameters.
            w = w - eta * v

        # Record the objective value and computation cost.
        problem.checkpoint(w)

    # Return the final parameters.
    return w

# Step 19 - saga
def saga(problem, w0, eta, n_steps, seed=0):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Create a floating-point copy of the initial parameters.
    w = np.array(w0, dtype=float, copy=True)

    # Initialize the gradient table using the starting parameters.
    # Each row stores the gradient for one training example.
    table = np.array(
        [problem.grad_i(w, n) for n in range(problem.N)],
        dtype=float
    )

    # Initialize the average of all stored gradients.
    g_avg = table.mean(axis=0)

    # Record the initial objective value without additional cost.
    problem.checkpoint(w)

    # Perform the specified number of SAGA updates.
    for t in range(n_steps):

        # Randomly select one training example.
        n = int(rng.integers(problem.N))

        # Calculate the new gradient for the selected example.
        g_new = problem.grad_i(w, n)

        # Calculate the SAGA variance-reduced gradient estimator.
        g = g_new - table[n] + g_avg

        # Update the average gradient using the difference
        # between the new and previously stored gradients.
        g_avg = g_avg + (g_new - table[n]) / problem.N

        # Replace the old gradient with the newly calculated gradient.
        table[n] = g_new

        # Update the model parameters.
        w = w - eta * g

        # Record a checkpoint after every N updates.
        if (t + 1) % problem.N == 0:
            problem.checkpoint(w)

    # Return the final parameters and gradient table.
    return w, table

# Step 20 - adagrad_update
def adagrad_update(theta, g, state, eta=0.1, eps=1e-6):
    # Convert inputs to floating-point NumPy arrays.
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(g, dtype=float)

    # Retrieve the previous accumulated squared gradients.
    # Initialize with zeros if this is the first update.
    s_old = np.asarray(
        state.get("s", np.zeros_like(g)),
        dtype=float
    )

    # Accumulate squared gradients.
    s = s_old + g**2

    # Apply the adaptive AdaGrad update.
    new_theta = theta - eta * g / np.sqrt(s + eps)

    # Return a new state dictionary without modifying the original.
    new_state = {
        "s": s
    }

    return new_theta, new_state


def rmsprop_update(theta, g, state, eta=0.001, beta=0.9, eps=1e-6):
    # Convert inputs to floating-point NumPy arrays.
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(g, dtype=float)

    # Retrieve the previous exponential moving average of squared gradients.
    s_old = np.asarray(
        state.get("s", np.zeros_like(g)),
        dtype=float
    )

    # Update the moving average of squared gradients.
    s = beta * s_old + (1.0 - beta) * g**2

    # Apply the RMSProp adaptive update.
    new_theta = theta - eta * g / np.sqrt(s + eps)

    # Return a new state dictionary.
    new_state = {
        "s": s
    }

    return new_theta, new_state


def adadelta_update(theta, g, state, eta=1.0, beta=0.9, eps=1e-6):
    # Convert inputs to floating-point NumPy arrays.
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(g, dtype=float)

    # Retrieve the previous moving average of squared gradients.
    s_old = np.asarray(
        state.get("s", np.zeros_like(g)),
        dtype=float
    )

    # Retrieve the previous moving average of squared updates.
    delta_old = np.asarray(
        state.get("delta", np.zeros_like(g)),
        dtype=float
    )

    # Update the moving average of squared gradients.
    s = beta * s_old + (1.0 - beta) * g**2

    # Calculate the parameter update using the OLD delta average.
    Delta = (
        -eta
        * np.sqrt(delta_old + eps)
        / np.sqrt(s + eps)
        * g
    )

    # Update the parameters.
    new_theta = theta + Delta

    # Update the moving average of squared parameter updates.
    delta = beta * delta_old + (1.0 - beta) * Delta**2

    # Return a new state dictionary.
    new_state = {
        "s": s,
        "delta": delta
    }

    return new_theta, new_state


def run_stochastic(problem, w0, update, n_epochs, batch_size=32, seed=0):
    # Initialize the random number generator.
    rng = np.random.default_rng(seed)

    # Create a floating-point copy of the initial parameters.
    w = np.array(w0, dtype=float, copy=True)

    # Initialize the optimizer state.
    state = {}

    # Perform the specified number of epochs.
    for epoch in range(n_epochs):

        # Generate a new random permutation for this epoch.
        perm = rng.permutation(problem.N)

        # Process the shuffled dataset in consecutive minibatches.
        for start in range(0, problem.N, batch_size):

            # Extract the current minibatch.
            chunk = perm[start:start + batch_size]

            # Calculate the minibatch gradient.
            g = problem.grad_i(w, chunk)

            # Apply the selected adaptive optimizer.
            w, state = update(w, g, state)

        # Record the objective value after each epoch.
        problem.checkpoint(w)

    # Return the final optimized parameters.
    return w

# Step 21 - adam_update
def adam_update(theta, g, state, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-6, bias_correct=True):
    # Convert inputs to floating-point NumPy arrays.
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(g, dtype=float)

    # Retrieve the previous iteration count.
    t = state.get("t", 0) + 1

    # Retrieve the previous first and second moments.
    # Initialize both with zeros if this is the first update.
    m_old = np.asarray(
        state.get("m", np.zeros_like(g)),
        dtype=float
    )

    s_old = np.asarray(
        state.get("s", np.zeros_like(g)),
        dtype=float
    )

    # Update the first moment (exponential moving average of gradients).
    m = beta1 * m_old + (1.0 - beta1) * g

    # Update the second moment (exponential moving average of squared gradients).
    s = beta2 * s_old + (1.0 - beta2) * g**2

    # Apply bias correction when enabled.
    if bias_correct:
        m_hat = m / (1.0 - beta1**t)
        s_hat = s / (1.0 - beta2**t)
    else:
        # Use the uncorrected moments.
        m_hat = m
        s_hat = s

    # Apply the Adam parameter update.
    # Epsilon is added outside the square root.
    new_theta = theta - eta * m_hat / (np.sqrt(s_hat) + eps)

    # Store the uncorrected moments in a new state dictionary.
    new_state = {
        "t": t,
        "m": m,
        "s": s
    }

    return new_theta, new_state

# Step 22 - reddi_game
def yogi_update(theta, g, state, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-6):
    # Convert inputs to floating-point NumPy arrays.
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(g, dtype=float)

    # Retrieve previous first and second moments.
    m_old = np.asarray(
        state.get("m", np.zeros_like(g)),
        dtype=float
    )

    s_old = np.asarray(
        state.get("s", np.zeros_like(g)),
        dtype=float
    )

    # Update the first moment using Adam's rule.
    m = beta1 * m_old + (1.0 - beta1) * g

    # Update the second moment using Yogi's rule.
    s = (
        s_old
        + (1.0 - beta2) * g**2 * np.sign(g**2 - s_old)
    )

    # Apply the parameter update without bias correction.
    new_theta = theta - eta * m / (np.sqrt(s) + eps)

    # Return a new state dictionary.
    new_state = {
        "m": m,
        "s": s
    }

    return new_theta, new_state


def amsgrad_update(theta, g, state, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-6):
    # Convert inputs to floating-point NumPy arrays.
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(g, dtype=float)

    # Retrieve previous first and second moments.
    m_old = np.asarray(
        state.get("m", np.zeros_like(g)),
        dtype=float
    )

    s_old = np.asarray(
        state.get("s", np.zeros_like(g)),
        dtype=float
    )

    # Retrieve the previous maximum second-moment estimate.
    s_max_old = np.asarray(
        state.get("s_max", np.zeros_like(g)),
        dtype=float
    )

    # Update the first moment using Adam's rule.
    m = beta1 * m_old + (1.0 - beta1) * g

    # Update the second moment using Adam's rule.
    s = beta2 * s_old + (1.0 - beta2) * g**2

    # Maintain the element-wise running maximum.
    s_max = np.maximum(s_max_old, s)

    # Apply the AMSGrad parameter update without bias correction.
    new_theta = theta - eta * m / (np.sqrt(s_max) + eps)

    # Return a new state dictionary.
    new_state = {
        "m": m,
        "s": s,
        "s_max": s_max
    }

    return new_theta, new_state


def adopt_update(theta, g, state, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-6):
    # Convert inputs to floating-point NumPy arrays.
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(g, dtype=float)

    # Retrieve the previous first moment.
    m_old = np.asarray(
        state.get("m", np.zeros_like(g)),
        dtype=float
    )

    # Retrieve the previous second moment.
    # On the first update, use g squared as specified.
    s_old = np.asarray(
        state.get("s", g**2),
        dtype=float
    )

    # Normalize the gradient using the previous second moment.
    normalized_g = g / np.sqrt(s_old + eps)

    # Update the first moment using the normalized gradient.
    m = beta1 * m_old + (1.0 - beta1) * normalized_g

    # Apply the ADOPT parameter update.
    new_theta = theta - eta * m

    # Update the second moment.
    s = beta2 * s_old + (1.0 - beta2) * g**2

    # Return a new state dictionary.
    new_state = {
        "m": m,
        "s": s
    }

    return new_theta, new_state


def reddi_game(update, T, C=1010.0, every=101, small=10.0, alpha=0.5):
    # Initialize the parameter on the interval [-1, 1].
    x = np.array([0.5], dtype=float)

    # Initialize the optimizer state.
    state = {}

    # Initialize cumulative regret.
    regret = 0.0

    # Simulate the online optimization problem.
    for t in range(1, T + 1):

        # Determine the gradient for the current round.
        if t % every == 1:
            c = C
        else:
            c = -small

        # Accumulate regret relative to the optimal fixed point x = -1.
        regret += c * x[0] + c

        # Perform the adaptive optimizer update.
        x, state = update(
            x,
            np.array([c], dtype=float),
            state,
            eta=alpha / np.sqrt(t)
        )

        # Project the parameter back onto the interval [-1, 1].
        x = np.clip(x, -1.0, 1.0)

    # Return the final parameter and average regret as Python floats.
    return float(x[0]), float(regret / T)

# Step 23 - projected_gd
def project_box(theta, lower, upper):
    # Convert theta to a floating-point NumPy array.
    theta = np.asarray(theta, dtype=float)

    # Project each element onto the interval [lower, upper].
    projected = np.clip(theta, lower, upper)

    return projected


def projected_gd(grad, theta0, lower, upper, eta, n_iter):
    # Create a floating-point copy of the initial parameters.
    # This ensures theta0 is not modified.
    theta = np.array(theta0, dtype=float, copy=True)

    # Project the initial point onto the feasible box.
    theta = project_box(theta, lower, upper)

    # Store all iterates, beginning with the projected initial point.
    path = np.empty((n_iter + 1, theta.size), dtype=float)
    path[0] = theta

    # Perform projected gradient descent.
    for t in range(n_iter):
        # Calculate the gradient at the current point.
        g = np.asarray(grad(theta), dtype=float)

        # Take a gradient descent step.
        theta = theta - eta * g

        # Project the updated parameters onto the feasible box.
        theta = project_box(theta, lower, upper)

        # Store the current iterate.
        path[t + 1] = theta

    return path

# Step 24 - active_set_qp
def kkt_residuals(H, c, A, b, theta, mu):
    # Convert all inputs to floating-point NumPy arrays.
    H = np.asarray(H, dtype=float)
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    theta = np.asarray(theta, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # Calculate constraint violations: A @ theta - b.
    violations = A @ theta - b

    # 1. Primal feasibility residual.
    primal = (
        max(0.0, float(np.max(violations)))
        if violations.size > 0
        else 0.0
    )

    # 2. Stationarity residual:
    # H @ theta + c + A.T @ mu = 0.
    stationarity_vector = H @ theta + c + A.T @ mu

    stationarity = (
        float(np.max(np.abs(stationarity_vector)))
        if stationarity_vector.size > 0
        else 0.0
    )

    # 3. Dual feasibility residual: mu >= 0.
    dual = (
        max(0.0, float(np.max(-mu)))
        if mu.size > 0
        else 0.0
    )

    # 4. Complementarity residual:
    # mu * (A @ theta - b) = 0.
    complementarity_vector = mu * violations

    complementarity = (
        float(np.max(np.abs(complementarity_vector)))
        if complementarity_vector.size > 0
        else 0.0
    )

    # Return all KKT residuals as Python floats.
    return {
        "primal": primal,
        "stationarity": stationarity,
        "dual": dual,
        "complementarity": complementarity
    }


def active_set_qp(H, c, A, b, theta0, max_iter=100, tol=1e-9):
    # Convert inputs to floating-point NumPy arrays.
    H = np.asarray(H, dtype=float)
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # Create a floating-point copy of the feasible starting point.
    theta = np.array(theta0, dtype=float, copy=True)

    # Initialize an empty working set.
    # It stores the indices of active constraints.
    working_set = []

    # Number of constraints.
    n_constraints = len(b)

    for _ in range(max_iter):

        # Calculate the gradient of the quadratic objective.
        grad = H @ theta + c

        # --------------------------------------------------
        # Step 1: Solve the equality-constrained QP system.
        # --------------------------------------------------

        if working_set:
            # Extract the rows corresponding to active constraints.
            A_W = A[working_set]

            # Number of active constraints.
            n_active = len(working_set)

            # Construct the KKT system.
            KKT = np.block([
                [H, A_W.T],
                [A_W, np.zeros((n_active, n_active))]
            ])

            # Construct the right-hand side.
            rhs = np.concatenate((
                -grad,
                np.zeros(n_active)
            ))

            # Solve for the search direction and working-set multipliers.
            solution = np.linalg.solve(KKT, rhs)

            p = solution[:len(theta)]
            mu_W = solution[len(theta):]

        else:
            # With an empty working set, solve the unconstrained
            # Newton direction system.
            p = np.linalg.solve(H, -grad)

            mu_W = np.empty(0, dtype=float)

        # --------------------------------------------------
        # Step 2: Check whether the search direction is zero.
        # --------------------------------------------------

        if np.linalg.norm(p) <= tol:

            # If the working set is empty or all multipliers
            # are sufficiently nonnegative, the solution is optimal.
            if not working_set or np.min(mu_W) >= -tol:

                # Construct the complete multiplier vector.
                mu = np.zeros(n_constraints, dtype=float)

                # Assign multipliers to the active constraints.
                if working_set:
                    mu[working_set] = mu_W

                return theta, mu

            # Otherwise, remove the constraint with the most
            # negative multiplier from the working set.
            remove_position = int(np.argmin(mu_W))
            del working_set[remove_position]

            # Recalculate the search direction.
            continue

        # --------------------------------------------------
        # Step 3: Calculate the maximum feasible step length.
        # --------------------------------------------------

        alpha = 1.0
        blocking_constraint = None

        # Identify constraints that are not currently active.
        working_set_indices = set(working_set)

        for i in range(n_constraints):

            # Skip constraints already in the working set.
            if i in working_set_indices:
                continue

            # Calculate the directional derivative of constraint i.
            a_i_p = np.dot(A[i], p)

            # Only constraints with positive directional derivative
            # can block movement along the search direction.
            if a_i_p > tol:

                # Calculate the maximum feasible step for constraint i.
                alpha_i = (
                    b[i] - np.dot(A[i], theta)
                ) / a_i_p

                # Identify the first blocking constraint.
                if alpha_i < alpha:
                    alpha = float(alpha_i)
                    blocking_constraint = i

        # --------------------------------------------------
        # Step 4: Update the current solution.
        # --------------------------------------------------

        theta = theta + alpha * p

        # --------------------------------------------------
        # Step 5: Add the blocking constraint if one exists.
        # --------------------------------------------------

        if blocking_constraint is not None:
            working_set.append(blocking_constraint)

    # Raise an error if the algorithm does not converge.
    raise RuntimeError(
        "Active-set QP solver did not converge within max_iter iterations."
    )

