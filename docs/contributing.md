# Contributing

See the repository `CONTRIBUTING.md` for commands and review standards.

Statistical changes should include a data-generating process with known truth, quantitative
recovery metrics, and tests for the failure mode being addressed. Optimization changes should
include likelihood agreement on small matrices and a scaling benchmark. AnnData changes must
preserve axis alignment and never overwrite user data by default.
