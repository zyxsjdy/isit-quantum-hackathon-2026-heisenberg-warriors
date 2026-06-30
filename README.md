# QAOA-ISAC UAV Deployment

This project formulates UAV placement for integrated sensing and communication
as a constrained quantum optimization problem. The deployment must assign UAVs
to candidate 3-D grid points while respecting one-hot assignment, no
co-location, survivor SINR, and collision-avoidance constraints.

The strongest result is a valid-subspace QAOA-guided candidate generator plus
classical local search. The valid-subspace solver keeps the search inside
feasible UAV assignments, ranks candidates by QAOA probability, and then uses
local search to polish only a small top-K candidate set.

The current benchmark also separates the raw QAOA ranking from the local-search
polishing step, adds a simulator-side probability-noise robustness probe, and
includes a larger `U=4, G=7, S=7` scale challenge.

## Result Snapshot

The current benchmark evidence is regenerated in
`qaoa_isac_benchmark_results.json`.

| Scenario | Method | AR rate | Notes |
| --- | --- | ---: | --- |
| Headline `U=4, G=6, S=6` | Exact enumeration | 1.000 | Feasible optimum reference |
| Headline `U=4, G=6, S=6` | Greedy | 0.811 | Feasible but 18.9% below optimum |
| Headline `U=4, G=6, S=6` | Valid-subspace QAOA top state | 1.000 | Matches exact optimum |
| Headline `U=4, G=6, S=6` | QAOA raw top-8 | 1.000 | No local-search gain needed |
| Headline `U=4, G=6, S=6` | QAOA top-8 + local search | 1.000 | Matches exact optimum |
| Headline `U=4, G=6, S=6` | Simulated annealing | 0.993 mean | 0.719 optimum-hit rate across trials |
| IBM `ibm_quebec` `U=4, G=6, S=6` | Raw feasible hardware sample | 0.829 best | 1/1024 raw samples were feasible |
| IBM `ibm_quebec` `U=4, G=6, S=6` | Projected hardware candidate | 1.000 best | Full-binary QUBO bridge plus feasible projection |
| Next hardware candidate `U=3, G=6, S=5` | QAOA top state | 1.000 | 18 qubits, 22 feasible assignments |
| Next hardware candidate `U=3, G=6, S=5` | Full-binary circuit | - | Depth 103, 306 CX gates before ISA transpilation |
| Multi-seed suite `U=3, G=6, S=5` | QAOA raw top-8 | 0.994 mean | 25 evaluated seeds |
| Multi-seed suite `U=3, G=6, S=5` | QAOA top-8 + local search | 0.999 mean | 23/25 optimum hits |
| Multi-seed suite `U=3, G=6, S=5` | Simulated annealing | 0.986 mean | 4/25 optimum hits |
| Stress seeds | QAOA top-4 + local search | 1.000 mean | 12/12 optimum hits |
| Stress seeds | Simulated annealing | 0.990 mean | 4/12 optimum hits |
| Scale challenge `U=4, G=7, S=7` | Greedy | 0.819 | 724 feasible assignments, 28 full-binary variables |
| Scale challenge `U=4, G=7, S=7` | QAOA raw top-8 | 0.927 | Larger exact-valid-subspace case |
| Scale challenge `U=4, G=7, S=7` | QAOA top-8 + local search | 1.000 | Matches exact optimum |
| Scale challenge `U=4, G=7, S=7` | Simulated annealing | 0.967 mean | Strong but below QAOA + local |

QAOA assigns 6.25x more probability to the headline optimum than uniform
feasible sampling. At a 95% target success probability, the headline optimum
requires 157 QAOA samples versus 985 uniform feasible samples.

## Candidate-Efficiency Evidence

The judging-relevant comparison is not just "QAOA can find a good answer"; it
is whether QAOA helps under the same candidate budget. The top-K sweep compares
QAOA-ranked candidates against random top-K multi-start local search. A
simulated annealing baseline is also included as a stronger classical heuristic.

Headline benchmark:

| K | QAOA AR | Random top-K + local AR | QAOA gain | Random optimum hit rate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 0.914 | 0.086 | 0.141 |
| 2 | 1.000 | 0.949 | 0.051 | 0.312 |
| 4 | 1.000 | 0.974 | 0.026 | 0.422 |
| 8 | 1.000 | 0.986 | 0.014 | 0.547 |
| 16 | 1.000 | 0.999 | 0.001 | 0.828 |

Stress suite:

| K | QAOA mean AR | Random mean AR | QAOA optimum hits |
| ---: | ---: | ---: | ---: |
| 1 | 0.934 | 0.874 | 4/12 |
| 2 | 0.964 | 0.920 | 6/12 |
| 4 | 1.000 | 0.965 | 12/12 |
| 8 | 1.000 | 0.992 | 12/12 |
| 16 | 1.000 | 0.999 | 12/12 |

This is the main technical claim: QAOA-guided candidate selection reaches the
optimum with fewer polished candidates on the stress cases where greedy is at
least 10% below exact optimum.

### Local-Search Attribution

The benchmark now records how much of the result comes from QAOA probability
ranking before local search.

| Scenario | QAOA raw top-K AR | QAOA local gain | Random raw top-K mean AR | Random local gain | Random top-K + local AR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Headline top-8 | 1.000 | 0.000 | 0.873 | 0.119 | 0.993 |
| Multi-seed suite top-8 | 0.994 mean | 0.006 mean | 0.926 mean | 0.065 mean | 0.990 mean |
| Stress suite top-8 | 0.994 mean | 0.006 mean | 0.913 mean | 0.078 mean | 0.990 mean |
| Scale challenge top-8 | 0.927 | 0.073 | 0.825 | 0.137 | 0.962 |

This reduces the local-search ambiguity: local search helps both QAOA and
random candidates, but QAOA starts from a stronger top-K candidate set.

### Probability-Noise Robustness Probe

This is a simulator-side stress test that blends the QAOA probability
distribution toward uniform feasible sampling. It is not a hardware noise
calibration.

| Uniform blend | Optimum probability | QAOA top-8 + local AR | Shots for 95% optimum sample |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.0190 | 1.000 | 157 |
| 0.10 | 0.0174 | 1.000 | 171 |
| 0.25 | 0.0150 | 1.000 | 199 |
| 0.50 | 0.0110 | 1.000 | 271 |
| 0.75 | 0.0070 | 1.000 | 425 |
| 1.00 | 0.0030 | 1.000 | 985 |

Even when the probability concentration is blended fully back to uniform,
top-8 local polishing still recovers the optimum on the headline case. The
sample-efficiency advantage, however, degrades toward uniform sampling as
expected.

## Limitation Status

| Limitation | Current status |
| --- | --- |
| No measured hardware result yet | Resolved. IBM job `d91i01vccmks73d56i80` on `ibm_quebec` completed with 1024 shots. Raw feasible sampling was 1/1024; feasible projection recovered the exact optimum. |
| Hardware path differs from headline valid-subspace solver | Still true. The hardware path remains a full-binary QUBO bridge and should not be claimed as the custom valid-subspace mixer. |
| Small scale | Reduced, not eliminated. The benchmark now includes a `U=4, G=7, S=7` scale challenge with 724 feasible assignments and 28 full-binary variables. |
| Local search may dominate | Reduced. The JSON and README now report raw QAOA top-K quality and local-search gain separately. |
| Simulated annealing is close | Kept explicit. SA remains a strong baseline, especially on stress cases, but trails QAOA + local in the recorded mean AR and optimum-hit metrics. |
| Noise robustness missing | Reduced with a probability-blend robustness probe and a real hardware sample. The hardware result shows noise is severe for raw feasibility, so projection/post-processing must be presented honestly. |
| `QAOA_ISAC_UAV.ipynb` correctness red flags | Quarantined as reference-only and ignored by git; it is not used as benchmark evidence. |

## Visual Evidence

The benchmark generates presentation-ready PNGs in `figures/`.

![QAOA convergence](figures/qaoa_convergence.png)

![QAOA angle landscape](figures/qaoa_angle_landscape.png)

![Candidate efficiency](figures/qaoa_candidate_efficiency.png)

![Probability distribution](figures/qaoa_probability_distribution.png)

![Deployment map](figures/qaoa_deployment_map.png)

## Judging Criteria Mapping

| Criterion | Evidence in this repo |
| --- | --- |
| Innovation and originality | Constrained QAOA-guided UAV deployment for ISAC with SINR and collision constraints. |
| Technical depth | Exact feasible reference, greedy baseline, QAOA probability distribution, top-K sample-efficiency sweep, local-search polishing, and multi-seed stress suite. |
| Feasibility | Reproducible Python benchmark, saved JSON results, notebook view, and guarded IBM Quantum Runtime hardware section. |
| Presentation | `qaoa_isac_benchmark.ipynb` presents the benchmark flow and `README.md` summarizes the win condition. |
| Teamwork | Add final named member contributions before submission once the team roster is confirmed. |

For the teamwork score, the final submission should name who owned the ISAC
model, QAOA solver, benchmark analysis, notebook/results, hardware path, and
presentation. Those names are not inferred in this README.

## Reproduce

Use the qiskit environment that was used to generate the checked-in results:

```powershell
& 'C:\Users\harry\.conda\envs\qiskit\python.exe' -X utf8 .\qaoa_isac_benchmark.py --include-suite --include-scale-challenge --include-hardware-demo-candidate --grid-steps 81 --random-trials 64 --sweep-random-trials 64 --sa-trials 32 --suite-random-trials 32 --suite-sweep-random-trials 32 --suite-sa-trials 16 --make-figures
```

Quick syntax check:

```powershell
& 'C:\Users\harry\.conda\envs\qiskit\python.exe' -X utf8 -m py_compile .\qaoa_isac_benchmark.py
```

## Files

| File | Purpose |
| --- | --- |
| `qaoa_isac_env.py` | Physical channel model, constraints, and QUBO construction |
| `qaoa_isac_notebook.ipynb` | Original end-to-end notebook, preserved |
| `qaoa_isac_benchmark.py` | Reproducible benchmark harness |
| `qaoa_isac_benchmark.ipynb` | Judge-facing notebook view |
| `qaoa_isac_benchmark_results.json` | Regenerated benchmark and sweep metrics |
| `figures/` | Generated convergence, landscape, probability, efficiency, and deployment plots |
| `qaoa_isac_figures.pdf` | Existing figure artifact |

## Hardware Status

The notebook includes a guarded IBM Quantum Runtime section with
`SUBMIT_TO_HARDWARE = False` to avoid accidental resubmission. It can retrieve
the completed job `d91i01vccmks73d56i80` from `ibm_quebec`.

The hardware circuit is a full-binary QUBO bridge with 24 qubits, not the
custom valid-subspace mixer used by the headline simulator. The completed job
used 1024 shots. After ISA transpilation the circuit depth was 1233 with 1572
CZ gates.

Raw hardware feasibility was low: only 1 of 1024 measured bitstrings was
already feasible, with AR 0.829. Feasible projection/post-processing recovered
the exact optimum assignment `[2, 4, 1, 3]` with AR 1.000 from the measured
samples. Present this as real-hardware execution evidence, not hardware
quantum-advantage evidence.

The next hardware-improvement target is a smaller `U=3, G=6, S=5`, seed `35`
case recorded in `hardware_demo_candidate`. It uses 18 full-binary qubits and
the pre-ISA circuit has depth 103 with 306 CX gates, down from the 24-qubit
headline hardware bridge with depth 139 and 552 CX gates before ISA
transpilation. On the valid-subspace simulator this candidate has greedy AR
0.785, QAOA top-state AR 1.000, and optimum probability 0.358.

See `docs/ibm-hardware-run.md` for the exact install, credential, and notebook
cells to change when submitting a hardware job.
