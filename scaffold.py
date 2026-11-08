"""
Optimization from Scratch in NumPy: From Steepest Descent to L-BFGS, SVRG and Adam scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Optimization from scratch: every solver in chapter 8 of Murphy's Probabilistic Machine Learning, on one ledger.

Runs the book's experiments in order, reports what reproduced, and finishes with the scoreboard: all ten smooth
solvers charged in passes over the data, on breast-cancer data and on a larger synthetic problem.
"""
import numpy as np


def rel_gap(value, f_star, gap0):
    return max(value - f_star, 0.0) / gap0


def first_order():
    print("1. First-order methods")
    print("   Eq. 8.19 is not convex: Hessian eigenvalues at (0, 1) are", np.round(np.linalg.eigvalsh(valley_hessian([0.0, 1.0])), 3).tolist())
    limit = max_stable_step(valley_hessian([1.0, 1.0]))
    print(f"   Figure 8.11: step limit at the optimum 2/lambda_max = {limit:.3f}")
    for eta in (0.1, 0.34, 0.35, 0.6):
        path = gd_fixed(valley_grad, [0.0, 0.0], eta, 2000)
        print(f"     eta = {eta:<4}: after 20 steps loss {valley_loss(path[20]):.4f}; distance from (1, 1) after 2000 steps {np.linalg.norm(path[-1] - 1.0):.4f}")
    path = steepest_descent_armijo(valley_loss, valley_grad, [0.0, 0.0], 100)
    print(f"     Armijo: loss {valley_loss(path[-1]):.2e} after 100 steps")
    print("   Figure 8.12, eq. 8.29:")
    for name, A in (("A = [20 5; 5 2]", A_SKEWED), ("A = [20 5; 5 16]", A_ROUND)):
        star = np.linalg.solve(A, -B_FIG)
        f_star = quadratic_loss(star, A, B_FIG)
        worst = steepest_descent_exact(A, B_FIG, worst_case_start(A, B_FIG), 3)
        easy = steepest_descent_exact(A, B_FIG, [0.0, 0.0], 3)
        r_worst = observed_rates([quadratic_loss(t, A, B_FIG) for t in worst], f_star)[-1]
        r_easy = observed_rates([quadratic_loss(t, A, B_FIG) for t in easy], f_star)[-1]
        print(f"     {name}: kappa {condition_number(A):.4f}, bound {rate_bound(condition_number(A)):.5f}, worst-case start {r_worst:.5f}, start (0, 0) {r_easy:.5f}")
    print("   Iterations to a 1e-8 gap on a 20-D quadratic:")
    for kappa in (100.0, 10000.0):
        A, b = make_quadratic(kappa, 20, 0)
        star = np.linalg.solve(A, -b)
        f = lambda t: quadratic_loss(t, A, b)
        g = lambda t: quadratic_grad(t, A, b)
        tm = tuned_momentum(1.0, kappa)
        gd = iters_to_gap(gd_fixed(g, np.zeros(20), 1.0 / kappa, int(8 * kappa)), f, f(star), 1e-8)
        hb = iters_to_gap(heavy_ball(g, np.zeros(20), *tm["heavy_ball"], 3000), f, f(star), 1e-8)
        ne = iters_to_gap(nesterov(g, np.zeros(20), *tm["nesterov"], 3000), f, f(star), 1e-8)
        print(f"     kappa {kappa:>8.0f}: GD 1/L {gd}, heavy ball {hb}, Nesterov {ne}")


def second_order():
    print("\n2. Second-order methods")
    f = lambda t: float(-t[0] ** 3 + 15 * t[0] ** 2)
    g = lambda t: np.array([-3 * t[0] ** 2 + 30 * t[0]])
    h = lambda t: np.array([[-6 * t[0] + 30]])
    print(f"   Figure 8.14(b), from x = 8: pure Newton ends at x = {newton_minimize(f, g, h, [8.0], 8, line_search=False)[-1][0]:.4f} (a maximum), trust region at x = {abs(trust_region_minimize(f, g, h, [8.0])[-1][0]):.4f} (the minimum)")
    path, C = bfgs_minimize(valley_loss, valley_grad, [0.0, 0.0])
    print(f"   BFGS on the valley: {len(path) - 1} iterations; C = {np.round(C, 2).tolist()}, true inverse Hessian {np.round(np.linalg.inv(valley_hessian([1.0, 1.0])), 2).tolist()}")
    X, y = load_cancer()
    ref = LogisticProblem(X, y, 1e-3)
    f_star = ref.value(newton_minimize(ref.loss, ref.grad, ref.hess, np.zeros(ref.D), 40, tol=1e-12)[-1])
    gap0 = ref.value(np.zeros(ref.D)) - f_star
    costs = []
    for m in (1, 3, 5, 10, 20):
        Q = LogisticProblem(X, y, 1e-3)
        lbfgs_minimize(Q.loss, Q.grad, np.zeros(Q.D), memory=m, n_iter=200)
        costs.append((m, passes_to_gap(Q, f_star, gap0, 1e-6)))
    print("   L-BFGS passes to a 1e-6 gap on breast-cancer data, by memory M:", costs)


def stochastic():
    print("\n3. Stochastic methods")
    X, y = lms_data()
    path = lms(X, y, [-0.5, 2.0], lambda t: 0.5 * 3 / (t + 3), 30)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    d2 = ((path[1:] - path[:-1]) ** 2).sum(axis=1)
    print(f"   Figure 8.16 (LMS): least squares {np.round(ols, 3).tolist()} (book: 1.45, 0.93); after 30 updates {np.round(path[-1], 3).tolist()}")
    print(f"     last update with a squared move >= 0.01: {int(np.nonzero(d2 >= 1e-2)[0][-1]) + 1} (book: about 26), MSE then {np.mean((X @ path[-1] - y) ** 2):.3f} against an optimum of {np.mean((X @ ols - y) ** 2):.4f}")
    print("   Eq. 8.61, ratio at T = 100, 10000:", {k: [round(robbins_monro_ratio(k, T), 3) for T in (100, 10000)] for k in ("constant", "exponential", "polynomial")})
    X, y = make_logistic(2000, 20, 0)
    ref = LogisticProblem(X, y, 1e-3)
    f_star = ref.value(newton_minimize(ref.loss, ref.grad, ref.hess, np.zeros(20), 40, tol=1e-12)[-1])
    gap0 = ref.value(np.zeros(20)) - f_star
    print("   SGD, 2,000 rows, batch 8, 60 epochs, eta0 = 0.3: relative gap of the last iterate and the average")
    for label, kind, kw in (("constant", "constant", {}), ("exponential, lam 1e-3", "exponential", dict(lam=1e-3)), ("square root", "polynomial", dict(beta=0.01)), ("step decay", "piecewise", dict(gamma=0.5, every=2500)), ("warmup-cosine", "warmup_cosine", dict(t_warm=500, t_total=15000))):
        Q = LogisticProblem(X, y, 1e-3)
        out = sgd_minibatch(Q, np.zeros(20), lambda t: lr_schedule(kind, t, eta0=0.3, **kw), 8, 60)
        print(f"     {label:24s} {rel_gap(Q.value(out['w']), f_star, gap0):.1e}   {rel_gap(Q.value(out['avg']), f_star, gap0):.1e}")
    Q = LogisticProblem(X, y, 1e-3)
    w = svrg(Q, np.zeros(20), 1.0 / max_smoothness(Q), 10)
    S = LogisticProblem(X, y, 1e-3)
    w2, _ = saga(S, np.zeros(20), 1.0 / (3 * max_smoothness(S)), 19 * S.N)
    print(f"   Constant step, same data: SVRG {rel_gap(Q.value(w), f_star, gap0):.1e} after {Q.passes:.0f} passes, SAGA {rel_gap(S.value(w2), f_star, gap0):.1e} after {S.passes:.0f} passes")
    kw = dict(beta1=0.9, beta2=0.99, eps=1e-8)
    print("   Reddi game (the best point is x = -1), x after T rounds:")
    for name, up in (("Adam", adam_update), ("AMSGrad", amsgrad_update), ("Yogi", yogi_update), ("ADOPT", adopt_update)):
        print(f"     {name:8s}", {T: round(reddi_game(lambda x, g, s, eta, up=up: up(x, g, s, eta=eta, **kw), T)[0], 3) for T in (10000, 40000)})


def constrained():
    print("\n4. Constraints and proximal methods")
    H, c = 2 * np.eye(2), -np.array([3.0, 0.25])
    A, b = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]), np.ones(4)
    theta, mu = active_set_qp(H, c, A, b, np.zeros(2))
    print(f"   Eq. 8.116: theta* = {np.round(theta + 0.0, 4).tolist()}, mu* = {np.round(mu + 0.0, 4).tolist()}, g = {np.round(A @ theta - b + 0.0, 4).tolist()}")
    print("   Klee-Minty pivots for D = 1..10:", [simplex_lp(*klee_minty(D))["pivots"] for D in range(1, 11)])
    X, y = load_cancer()
    ys = 2 * y - 1
    w_i, w_s = ista(X, ys, 0.05, 5000), subgradient_lasso(X, ys, 0.05, 5000)
    print(f"   Lasso, lam = 0.05: ISTA has {int((w_i != 0).sum())} non-zero weights of 31, the subgradient method {int((w_s != 0).sum())}")
    import itertools
    Xs = X[:, [0, 1, 2, 3, 5, 8, 9, 22, 23, 25, 28]]
    P = LogisticProblem(Xs, y, 0.0)
    cand = np.array(list(itertools.product([-1.0, 1.0], repeat=11)))
    best = float(np.logaddexp(0.0, -(2 * y - 1)[:, None] * (Xs @ cand.T)).mean(axis=0).min())

    def score(train):
        return float(np.mean([P.value(quantize(train(np.random.default_rng(s).normal(0.0, 0.5, 11)))) for s in range(20)]))

    print(f"   Binary weights, 11 features, 20 starts, brute-force best {best:.4f}: BinaryConnect {score(lambda t: binary_connect(P.grad, t, 0.3, 500)):.4f}, "
          f"ProxQuant slow ramp {score(lambda t: prox_quant_train(P.grad, t, 0.3, lambda k: 1e-5 * k, 500)):.4f}, fast ramp {score(lambda t: prox_quant_train(P.grad, t, 0.3, lambda k: 1e-3 * k, 500)):.4f}")


def em():
    print("\n5. EM")
    X = load_faithful()
    pi0, mu0, S0 = np.array([0.5, 0.5]), np.array([[-1.0, 1.0], [1.0, -1.0]]), np.array([np.eye(2), np.eye(2)])
    res = em_gmm(X, pi0, mu0, S0, 150)
    raw = np.column_stack([np.array(FAITHFUL_DURATION, dtype=float), np.array(FAITHFUL_WAITING, dtype=float)])
    ll = np.array(res["ll"])
    print(f"   Figure 8.25: within 1e-3 of the final log likelihood after {int(np.nonzero(ll >= ll[-1] - 1e-3)[0][0])} iterations; monotone: {bool(np.all(np.diff(ll) >= -1e-9))}")
    print(f"     pi {np.round(res['pi'], 3).tolist()}, raw means {np.round(res['mu'] * raw.std(axis=0) + raw.mean(axis=0), 2).tolist()}, raw log likelihood {ll[-1] - 272 * np.log(raw.std(axis=0)).sum():.2f}")
    good = em_gmm_map(X, pi0, mu0, S0, 100)
    bad = em_gmm_map(X, pi0, mu0, S0, 12, literal_book=True)
    print(f"   Eq. 8.173 as printed: objective {bad['objective'][0]:.0f} -> {bad['objective'][-1]:.0f}, Sigma[0, 0] {bad['Sigma'][0][0, 0]:.4f}. Corrected: objective ends at {good['objective'][-1]:.2f}, monotone, Sigma[0, 0] {good['Sigma'][0][0, 0]:.4f}")
    print("   Figure 8.26(b), N = 100, K = 3, 20 trials: fraction failed (D, MLE, MAP)")
    print("    ", [(D, em_failure_rate(D, False, trials=20), em_failure_rate(D, True, trials=20)) for D in (5, 10, 20, 30, 40, 60, 80, 100)])


def scoreboard():
    print("\n6. The scoreboard: passes over the data to a 1e-6 relative gap, and the gap at the budget")
    X, y = load_cancer()
    data = (("Breast cancer, N = 569, D = 31, budget 300, theory steps for SVRG and SAGA", X, y, 300.0, 1.0),
            ("Breast cancer, same, SVRG and SAGA steps times 9", X, y, 300.0, 9.0),
            ("Synthetic, N = 5000, D = 20, budget 40", *make_logistic(5000, 20, 1), 40.0, 1.0))
    for label, Xd, yd, budget, scale in data:
        table = solver_race(Xd, yd, 1e-3, budget, vr_scale=scale)["table"]
        print("  ", label)
        for name, row in sorted(table.items(), key=lambda kv: (kv[1]["passes_to_tol"] is None, kv[1]["passes_to_tol"] or 0.0, kv[1]["gap_at_budget"])):
            reached = "-" if row["passes_to_tol"] is None else f"{row['passes_to_tol']:.0f}"
            print(f"     {name:18s} {reached:>5s}   {row['gap_at_budget']:.1e}")
    print("   Small N: quasi-Newton wins. Large N: SAGA and SVRG overtake it. SGD and Adam are quick early, then stall.")
    print("   Step sizes for the first-order and stochastic methods come from theory or a default; tuning them is not on the bill.")


def main() -> None:
    first_order()
    second_order()
    stochastic()
    constrained()
    em()
    scoreboard()


if __name__ == "__main__":
    main()

