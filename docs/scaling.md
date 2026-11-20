# Scaling and benchmarks

| Operation | Time | Main additional memory |
|---|---:|---:|
| Naive/corrected covariance | $O(CG^2)$ | $O(G^2)$ |
| Eigenvalue PSD repair | $O(G^3)$ | $O(G^2)$ |
| Dense heteroscedastic MLE | $O(ICG^3)$ | $O(CG^2)$ in the current profiler |
| Factor ECM | roughly $O(ICGK^2)$ | $O(CG+CK^2+GK)$ |

$I$ is the number of iterations and $K\ll G$. The dense likelihood is intentionally
limited to moderate $G$. The factor estimator initializes with truncated SVD and keeps its
covariance factored until `covariance_` is requested.

`benchmarks/benchmark_estimators.py` defines representative cases at `(1000, 50)`,
`(5000, 100)`, and `(10000, 500)`. It omits dense MLE when the requested dimension would be
misleading or infeasible. Runtime depends strongly on BLAS, CPU, and the distribution of error
bars, so benchmark results are generated locally rather than advertised as universal numbers.
