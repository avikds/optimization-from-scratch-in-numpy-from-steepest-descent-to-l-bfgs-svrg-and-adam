# Optimization from Scratch in NumPy: From Steepest Descent to L-BFGS, SVRG and Adam

Murphy's Probabilistic Machine Learning has a tour of the solvers behind every model in the book. This project builds all of them in NumPy and makes them compete under one rule: every solver pays in passes over the data, and a ledger records what it bought. You start on the book's test bed: quadratics with a chosen condition number, the book's valley function (which the book calls convex and which is not), and L2 logistic regression on breast-cancer data. Then fixed steps and the 2/lambda_max limit, exact and Armijo line search, the book's rate formula hit to five digits, heavy-ball and Nesterov momentum, Newton's method walking to a maximum, trust regions, BFGS with a Wolfe line search, and L-BFGS through the two-loop recursion. Next come LMS, learning-rate schedules and Robbins-Monro, Polyak-Ruppert averaging, SVRG and SAGA; AdaGrad, RMSProp, AdaDelta, Adam, Yogi, AMSGrad and ADOPT, with Adam ending at the worst point of the Reddi counterexample; projected gradient, an active-set QP with KKT certificates, a simplex solver that really does take 2^D - 1 pivots on Klee-Minty cubes, ISTA for the lasso, and BinaryConnect versus ProxQuant against a brute-force optimum. Then EM on Old Faithful, the collapsing variance, and MAP-EM surviving where MLE crashes. Headline numbers: on breast-cancer data L-BFGS reaches a 1e-6 relative gap in 32 passes, BFGS in 87, Newton in 257, and SAGA and SVRG need 187 and 204 even after step tuning; on 5,000 synthetic rows the order flips, with SAGA at 10 passes, SVRG at about 20 and L-BFGS at 23. Constant-step SGD on the same kind of data stalls between 5e-3 and 2e-2, and plain iterate averaging brings it to 4e-4. EM on Old Faithful lands on the textbook fit (log likelihood -1130.26), and on the book's MAP-versus-MLE experiment MLE-EM fails 50 to 75 percent of the time at D = 20 and always by D = 40, while MAP-EM never fails. Along the way you find what the printed equations get wrong, including the MAP covariance update, whose literal form drives the objective from -1,020 to -8,450 on Old Faithful.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_quadratic
- [x] **2.** valley_loss
- [x] **3.** LogisticProblem
- [x] **4.** gd_fixed
- [x] **5.** steepest_descent_exact
- [x] **6.** steepest_descent_armijo
- [x] **7.** heavy_ball
- [x] **8.** newton_minimize

---

Built on Deep-ML.
