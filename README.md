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
- [x] **9.** trust_region_minimize
- [x] **10.** bfgs_update_inverse
- [x] **11.** wolfe_line_search
- [x] **12.** bfgs_minimize
- [x] **13.** lbfgs_direction
- [x] **14.** lbfgs_minimize
- [x] **15.** lms
- [x] **16.** lr_schedule
- [x] **17.** sgd_minibatch
- [x] **18.** svrg
- [x] **19.** saga
- [x] **20.** adagrad_update
- [x] **21.** adam_update
- [x] **22.** reddi_game
- [x] **23.** projected_gd
- [x] **24.** active_set_qp
- [x] **25.** simplex_lp
- [x] **26.** ista
- [x] **27.** prox_quant
- [x] **28.** em_gmm
- [x] **29.** em_gmm_map
- [x] **30.** solver_race

## Results

```
1. First-order methods
   Eq. 8.19 is not convex: Hessian eigenvalues at (0, 1) are [-1.0, 1.0]
   Figure 8.11: step limit at the optimum 2/lambda_max = 0.343
     eta = 0.1 : after 20 steps loss 0.0621; distance from (1, 1) after 2000 steps 0.0000
     eta = 0.34: after 20 steps loss 0.0061; distance from (1, 1) after 2000 steps 0.0000
     eta = 0.35: after 20 steps loss 0.0056; distance from (1, 1) after 2000 steps 0.0680
     eta = 0.6 : after 20 steps loss 0.5244; distance from (1, 1) after 2000 steps 0.8047
     Armijo: loss 2.20e-08 after 100 steps
   Figure 8.12, eq. 8.29:
     A = [20 5; 5 2]: kappa 30.2336, bound 0.87603, worst-case start 0.87603, start (0, 0) 0.38571
     A = [20 5; 5 16]: kappa 1.8538, bound 0.08951, worst-case start 0.08951, start (0, 0) 0.01338
   Iterations to a 1e-8 gap on a 20-D quadratic:
     kappa      100: GD 1/L 644, heavy ball 48, Nesterov 88
     kappa    10000: GD 1/L 69353, heavy ball 506, Nesterov 927

2. Second-order methods
   Figure 8.14(b), from x = 8: pure Newton ends at x = 10.0000 (a maximum), trust region at x = 0.0000 (the minimum)
   BFGS on the valley: 10 iterations; C = [[1.01, 2.04], [2.04, 5.11]], true inverse Hessian [[1.0, 2.0], [2.0, 5.0]]
   L-BFGS passes to a 1e-6 gap on breast-cancer data, by memory M: [(1, 59.0), (3, 34.0), (5, 33.0), (10, 32.0), (20, 25.0)]

3. Stochastic methods
   Figure 8.16 (LMS): least squares [1.484, 0.905] (book: 1.45, 0.93); after 30 updates [1.235, 1.069]
     last update with a squared move >= 0.01: 17 (book: about 26), MSE then 0.076 against an optimum of 0.0085
   Eq. 8.61, ratio at T = 100, 10000: {'constant': [1.0, 1.0], 'exponential': [0.525, 0.525], 'polynomial': [0.279, 0.049]}
   SGD, 2,000 rows, batch 8, 60 epochs, eta0 = 0.3: relative gap of the last iterate and the average
     constant                 2.1e-02   3.6e-04
     exponential, lam 1e-3    1.2e-02   1.5e-02
     square root              1.2e-03   7.8e-03
     step decay               1.4e-05   6.0e-04
     warmup-cosine            2.9e-06   8.3e-04
   Constant step, same data: SVRG 4.7e-11 after 30 passes, SAGA 8.0e-08 after 20 passes
   Reddi game (the best point is x = -1), x after T rounds:
     Adam     {10000: 0.996, 40000: 0.994}
     AMSGrad  {10000: -0.801, 40000: -0.89}
     Yogi     {10000: -0.684, 40000: -0.695}
     ADOPT    {10000: -0.971, 40000: -0.993}

4. Constraints and proximal methods
   Eq. 8.116: theta* = [1.0, 0.0], mu* = [0.625, 0.375, 0.0, 0.0], g = [0.0, 0.0, -2.0, -2.0]
   Klee-Minty pivots for D = 1..10: [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023]
   Lasso, lam = 0.05: ISTA has 9 non-zero weights of 31, the subgradient method 31
   Binary weights, 11 features, 20 starts, brute-force best 0.1184: BinaryConnect 0.1542, ProxQuant slow ramp 0.1299, fast ramp 0.2674

5. EM
   Figure 8.25: within 1e-3 of the final log likelihood after 49 iterations; monotone: True
     pi [0.356, 0.644], raw means [[2.04, 54.48], [4.29, 79.97]], raw log likelihood -1130.26
   Eq. 8.173 as printed: objective -1020 -> -8451, Sigma[0, 0] 0.0034. Corrected: objective ends at -361.49, monotone, Sigma[0, 0] 0.0542
   Figure 8.26(b), N = 100, K = 3, 20 trials: fraction failed (D, MLE, MAP)
     [(5, 0.0, 0.0), (10, 0.25, 0.0), (20, 0.7, 0.0), (30, 0.9, 0.0), (40, 1.0, 0.0), (60, 1.0, 0.0), (80, 1.0, 0.0), (100, 1.0, 0.0)]

6. The scoreboard: passes over the data to a 1e-6 relative gap, and the gap at the budget
   Breast cancer, N = 569, D = 31, budget 300, theory steps for SVRG and SAGA
     L-BFGS(10)            32   5.3e-14
     BFGS                  87   1.8e-12
     Newton               257   0.0e+00
     GD, Armijo warm        -   2.8e-06
     Adam                   -   1.7e-05
     SGD, averaged          -   2.7e-05
     SVRG                   -   8.6e-04
     SAGA                   -   8.8e-04
     Nesterov               -   3.3e-03
     GD, step 1/L           -   3.9e-02
   Breast cancer, same, SVRG and SAGA steps times 9
     L-BFGS(10)            32   5.3e-14
     BFGS                  87   1.8e-12
     SAGA                 187   1.6e-08
     SVRG                 204   3.0e-08
     Newton               257   0.0e+00
     GD, Armijo warm        -   2.8e-06
     Adam                   -   1.7e-05
     SGD, averaged          -   2.7e-05
     Nesterov               -   3.3e-03
     GD, step 1/L           -   3.9e-02
   Synthetic, N = 5000, D = 20, budget 40
     SAGA                  10   0.0e+00
     SVRG                  21   1.7e-14
     L-BFGS(10)            23   1.1e-11
     SGD, averaged          -   1.2e-04
     Adam                   -   1.2e-04
     BFGS                   -   1.3e-03
     GD, Armijo warm        -   2.1e-03
     Nesterov               -   2.8e-02
     GD, step 1/L           -   1.4e-01
     Newton                 -   2.3e-01
   Small N: quasi-Newton wins. Large N: SAGA and SVRG overtake it. SGD and Adam are quick early, then stall.
   Step sizes for the first-order and stochastic methods come from theory or a default; tuning them is not on the bill.
```
