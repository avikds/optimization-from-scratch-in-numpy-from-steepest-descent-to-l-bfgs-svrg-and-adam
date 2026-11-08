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

