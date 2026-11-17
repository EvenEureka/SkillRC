# Calibration of the primary-metric decision

Simulated paired outcomes with a known true effect; margin δ = 0.05; 2000 bootstrap draws per replicate (the gate uses 20,000 with the same order-statistic rule); one-sided α = 0.025. Nominal targets: coverage 0.95; P(PASS) ≤ 0.025 at true Δ = −δ; P(BLOCK) ≤ 0.025 at true Δ = −δ.

Brackets: Monte Carlo 95% Wilson intervals over the replicates.

| scenario | n | strata | true Δ (mean) | coverage | P(PASS) | P(BLOCK) | P(INCONCL.) |
|---|---:|---|---:|---:|---:|---:|---:|
| binary (mean 0.5), A/A | 50 | single | +0.000 | 0.948 | 0.105 [0.086, 0.128] | 0.004 [0.001, 0.011] | 0.891 |
| webshop-like, A/A | 50 | single | -0.000 | 0.931 | 0.160 [0.136, 0.187] | 0.001 [0.000, 0.007] | 0.839 |
| binary (mean 0.5), boundary -delta | 50 | single | -0.050 | 0.935 | 0.039 [0.027, 0.054] | 0.026 [0.017, 0.040] | 0.935 |
| webshop-like, boundary -delta | 50 | single | -0.050 | 0.938 | 0.036 [0.025, 0.052] | 0.026 [0.017, 0.040] | 0.938 |
| binary (mean 0.5), regression -2delta | 50 | single | -0.100 | 0.930 | 0.005 [0.002, 0.013] | 0.113 [0.092, 0.136] | 0.882 |
| webshop-like, regression -2delta | 50 | single | -0.100 | 0.940 | 0.003 [0.001, 0.009] | 0.190 [0.164, 0.219] | 0.807 |
| binary (mean 0.5), A/A | 50 | D-like | +0.000 | 0.926 | 0.116 [0.096, 0.140] | 0.006 [0.003, 0.015] | 0.877 |
| webshop-like, A/A | 50 | D-like | -0.000 | 0.935 | 0.184 [0.158, 0.212] | 0.004 [0.001, 0.011] | 0.812 |
| binary (mean 0.5), boundary -delta | 50 | D-like | -0.050 | 0.943 | 0.028 [0.018, 0.041] | 0.029 [0.019, 0.043] | 0.944 |
| webshop-like, boundary -delta | 50 | D-like | -0.050 | 0.926 | 0.041 [0.029, 0.057] | 0.033 [0.022, 0.047] | 0.926 |
| binary (mean 0.5), regression -2delta | 50 | D-like | -0.100 | 0.921 | 0.005 [0.002, 0.013] | 0.139 [0.117, 0.164] | 0.856 |
| webshop-like, regression -2delta | 50 | D-like | -0.100 | 0.943 | 0.003 [0.001, 0.009] | 0.203 [0.176, 0.232] | 0.795 |
| rare-event mixture (round-4 counterexample) | 50 | single | -0.046 | 0.226 | 0.000 [0.000, 0.005] | 0.000 [0.000, 0.005] | 1.000 |
| binary (mean 0.5), A/A | 200 | single | +0.000 | 0.954 | 0.296 [0.266, 0.329] | 0.000 [0.000, 0.005] | 0.704 |
| webshop-like, A/A | 200 | single | -0.000 | 0.948 | 0.496 [0.462, 0.531] | 0.000 [0.000, 0.005] | 0.504 |
| binary (mean 0.5), boundary -delta | 200 | single | -0.050 | 0.939 | 0.029 [0.019, 0.043] | 0.028 [0.018, 0.041] | 0.944 |
| webshop-like, boundary -delta | 200 | single | -0.050 | 0.950 | 0.021 [0.013, 0.034] | 0.029 [0.019, 0.043] | 0.950 |
| binary (mean 0.5), regression -2delta | 200 | single | -0.100 | 0.943 | 0.000 [0.000, 0.005] | 0.306 [0.275, 0.339] | 0.694 |
| webshop-like, regression -2delta | 200 | single | -0.100 | 0.940 | 0.000 [0.000, 0.005] | 0.471 [0.437, 0.506] | 0.529 |
| binary (mean 0.5), A/A | 200 | D-like | +0.000 | 0.950 | 0.304 [0.273, 0.337] | 0.000 [0.000, 0.005] | 0.696 |
| webshop-like, A/A | 200 | D-like | -0.000 | 0.958 | 0.499 [0.464, 0.533] | 0.000 [0.000, 0.005] | 0.501 |
| binary (mean 0.5), boundary -delta | 200 | D-like | -0.050 | 0.936 | 0.030 [0.020, 0.044] | 0.034 [0.023, 0.049] | 0.936 |
| webshop-like, boundary -delta | 200 | D-like | -0.050 | 0.951 | 0.021 [0.013, 0.034] | 0.028 [0.018, 0.041] | 0.951 |
| binary (mean 0.5), regression -2delta | 200 | D-like | -0.100 | 0.953 | 0.000 [0.000, 0.005] | 0.314 [0.283, 0.347] | 0.686 |
| webshop-like, regression -2delta | 200 | D-like | -0.100 | 0.944 | 0.000 [0.000, 0.005] | 0.475 [0.441, 0.510] | 0.525 |
| rare-event mixture (round-4 counterexample) | 200 | single | -0.046 | 0.619 | 0.000 [0.000, 0.005] | 0.000 [0.000, 0.005] | 1.000 |
| binary (mean 0.5), A/A | 500 | single | +0.000 | 0.958 | 0.672 [0.633, 0.708] | 0.000 [0.000, 0.006] | 0.328 |
| webshop-like, A/A | 500 | single | -0.000 | 0.952 | 0.888 [0.861, 0.911] | 0.000 [0.000, 0.006] | 0.112 |
| binary (mean 0.5), boundary -delta | 500 | single | -0.050 | 0.960 | 0.018 [0.010, 0.033] | 0.018 [0.010, 0.033] | 0.963 |
| webshop-like, boundary -delta | 500 | single | -0.050 | 0.933 | 0.038 [0.026, 0.057] | 0.028 [0.018, 0.045] | 0.933 |
| binary (mean 0.5), regression -2delta | 500 | single | -0.100 | 0.958 | 0.000 [0.000, 0.006] | 0.648 [0.609, 0.685] | 0.352 |
| webshop-like, regression -2delta | 500 | single | -0.100 | 0.952 | 0.000 [0.000, 0.006] | 0.870 [0.841, 0.895] | 0.130 |
| binary (mean 0.5), A/A | 500 | D-like | +0.000 | 0.952 | 0.630 [0.591, 0.668] | 0.000 [0.000, 0.006] | 0.370 |
| webshop-like, A/A | 500 | D-like | -0.000 | 0.962 | 0.877 [0.848, 0.901] | 0.000 [0.000, 0.006] | 0.123 |
| binary (mean 0.5), boundary -delta | 500 | D-like | -0.050 | 0.933 | 0.030 [0.019, 0.047] | 0.035 [0.023, 0.053] | 0.935 |
| webshop-like, boundary -delta | 500 | D-like | -0.050 | 0.940 | 0.028 [0.018, 0.045] | 0.032 [0.020, 0.049] | 0.940 |
| binary (mean 0.5), regression -2delta | 500 | D-like | -0.100 | 0.955 | 0.000 [0.000, 0.006] | 0.652 [0.613, 0.689] | 0.348 |
| webshop-like, regression -2delta | 500 | D-like | -0.100 | 0.952 | 0.000 [0.000, 0.006] | 0.873 [0.844, 0.898] | 0.127 |
| rare-event mixture (round-4 counterexample) | 500 | single | -0.046 | 0.897 | 0.003 [0.001, 0.012] | 0.000 [0.000, 0.006] | 0.997 |

## Token guardrail (relative mean, tolerance +30%, 20,000 draws, n = 500, D-like strata)

| scenario | coverage | P(guardrail PASS) | P(guardrail BLOCK) | P(INCONCL.) |
|---|---:|---:|---:|---:|
| token guardrail, true change = tolerance (+30%) | 0.957 | 0.037 [0.021, 0.065] | 0.007 [0.002, 0.024] | 0.957 |
| token guardrail, true change +20% | 0.937 | 1.000 [0.987, 1.000] | 0.000 [0.000, 0.013] | 0.000 |
| token guardrail, no change | 0.927 | 1.000 [0.987, 1.000] | 0.000 [0.000, 0.013] | 0.000 |

Reading: the nominal one-sided level is 0.025. Where a Monte Carlo interval lies above it, the bootstrap
decision is anti-conservative for that scenario; the gate's error budget is nominal, not guaranteed.
The rare-event mixture shows poor interval coverage; the degeneracy rule turns those samples into
INCONCLUSIVE rather than wrong decisions.

Generated by `scripts/gate/calibration.py`.
