# QAOA-ISAC UAV Deployment

This project formulates UAV placement for integrated sensing and communication
as a constrained quantum optimization problem. The deployment must assign UAVs
to candidate 3-D grid points while respecting one-hot assignment, no
co-location, survivor SINR, and collision-avoidance constraints.

The strongest result is a candidate-aware valid-subspace QAOA candidate
generator plus classical local search. The valid-subspace solver keeps the
search inside feasible UAV assignments, trains angles in simulation, ranks
candidates by QAOA probability, and then uses local search to polish only a
small top-K candidate set.

The current evidence separates raw QAOA ranking from local-search polishing,
adds a multi-seed competition suite, includes a larger `U=4, G=7, S=7` scale
challenge, and keeps IBM hardware execution evidence separate from the
simulation headline.

## Result Snapshot

The current competition evidence is exported from
`qaoa_isac_training_notebook.ipynb` into `qaoa_isac_training_results.json`.
`qaoa_isac_submission_evidence.json` contains the distilled final-submission
numbers.

| Scenario | Method | AR rate | Notes |
| --- | --- | ---: | --- |
| Headline `U=4, G=6, S=6` | Exact enumeration | 1.000 | Feasible optimum reference |
| Headline `U=4, G=6, S=6` | Greedy | 0.811 | Feasible but 18.9% below optimum |
| Headline `U=4, G=6, S=6` | Expected-QAOA raw top-8 | 1.000 | Optimum already in raw QAOA candidates |
| Headline `U=4, G=6, S=6` | Candidate-aware QAOA top-8 + local | 1.000 | Matches exact optimum |
| Headline `U=4, G=6, S=6` | Simulated annealing | 0.993 mean | Strong classical heuristic baseline |
| Scale challenge `U=4, G=7, S=7` | Greedy + local | 0.835 | 724 feasible assignments, 28 full-binary variables |
| Scale challenge `U=4, G=7, S=7` | Expected-QAOA raw top-8 | 0.927 | Larger exact-valid-subspace case |
| Scale challenge `U=4, G=7, S=7` | Candidate-aware QAOA raw top-8 | 1.000 | Optimum appears before local polishing |
| Scale challenge `U=4, G=7, S=7` | Candidate-aware QAOA top-8 + local | 1.000 | Matches exact optimum |
| Scale challenge `U=4, G=7, S=7` | Simulated annealing | 0.968 mean | Strong but below QAOA + local |
| Multi-seed suite `U=3, G=6, S=5` | Greedy + local | 0.912 mean | 25 feasible evaluated cases from seeds `1:60` |
| Multi-seed suite `U=3, G=6, S=5` | Expected-QAOA raw top-8 | 0.994 mean | Top-K candidate quality before local search |
| Multi-seed suite `U=3, G=6, S=5` | Candidate-aware QAOA raw top-8 | 1.000 mean | Ties/beats expected-QAOA raw top-8 on 25/25 cases |
| Multi-seed suite `U=3, G=6, S=5` | Candidate-aware QAOA top-8 + local | 1.000 mean | 25/25 optimum hits |
| Multi-seed suite `U=3, G=6, S=5` | Random top-8 + local | 0.990 mean | Candidate-aware beats random mean on 14/25 cases |
| Multi-seed suite `U=3, G=6, S=5` | Simulated annealing | 0.986 mean | Strong classical heuristic baseline |
| Stress seeds | Candidate-aware QAOA top-8 + local | 1.000 mean | 12/12 optimum hits where greedy is at least 10% below exact |
| Stress seeds | Random top-8 + local | 0.990 mean | Candidate-aware beats random mean on 8/12 stress cases |
| Stress seeds | Simulated annealing | 0.986 mean | Below candidate-aware QAOA + local |
| IBM `ibm_quebec` `U=4, G=6, S=6` | Raw feasible hardware sample | 0.829 best | 1/1024 raw samples were feasible |
| IBM `ibm_quebec` `U=4, G=6, S=6` | Projected hardware candidate | 1.000 best | 16/1024 projected optimum count; below random-bitstring projection baseline |
| IBM `ibm_quebec` `U=3, G=6, S=5` | Projected hardware candidate | 1.000 best | 18-qubit job `d91ttqmu9n7c73ane4jg`; 203/1024 projected optimum count |
| IBM `ibm_quebec` `U=3, G=6, S=5` | Hardware/random projection ratio | 1.019 | Slightly above random projection baseline, with 0/1024 raw feasible samples |

On the scale challenge, the candidate-aware objective raises the raw top-8 AR
from 0.927 to 1.000 and increases optimum probability from 0.117% to 0.405%.
Across the multi-seed suite, candidate-aware QAOA places the optimum in the
top-8 candidate set for every feasible evaluated case.

## Candidate-Efficiency Evidence

The judging-relevant comparison is not just "QAOA can find a good answer"; it
is whether QAOA helps under the same candidate budget. The top-K sweep compares
candidate-aware QAOA-ranked candidates against random top-K multi-start local
search. A simulated annealing baseline is also included as a stronger classical
heuristic.

Multi-seed suite:

| K | Candidate-aware QAOA AR | Random top-K + local AR | QAOA gain | QAOA optimum hits |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.927 | 0.882 | 0.045 | 11/25 |
| 2 | 0.969 | 0.926 | 0.043 | 14/25 |
| 4 | 0.992 | 0.970 | 0.023 | 22/25 |
| 8 | 1.000 | 0.991 | 0.009 | 25/25 |
| 16 | 1.000 | 0.999 | 0.001 | 25/25 |

Stress suite:

| K | Candidate-aware QAOA AR | Random top-K + local AR | QAOA gain | QAOA optimum hits |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.910 | 0.866 | 0.044 | 4/12 |
| 2 | 0.964 | 0.922 | 0.042 | 6/12 |
| 4 | 0.985 | 0.967 | 0.018 | 10/12 |
| 8 | 1.000 | 0.991 | 0.009 | 12/12 |
| 16 | 1.000 | 1.000 | 0.000 | 12/12 |

This is the main technical claim: candidate-aware QAOA-guided candidate
selection reaches the optimum with a small polished candidate set, including
12/12 stress cases where greedy is at least 10% below exact optimum.

### Local-Search Attribution

The benchmark records how much of the result comes from QAOA probability
ranking before local search.

| Scenario | Expected-QAOA raw top-K AR | Candidate-aware raw top-K AR | Candidate-aware local gain | Random top-K + local AR |
| --- | ---: | ---: | ---: | ---: |
| Headline top-8 | 1.000 | 1.000 | 0.000 | 0.986 |
| Scale challenge top-8 | 0.927 | 1.000 | 0.000 | 0.959 |
| Multi-seed suite top-8 | 0.994 mean | 1.000 mean | 0.000 mean | 0.990 mean |
| Stress suite top-8 | 0.994 mean | 1.000 mean | 0.000 mean | 0.990 mean |

This reduces the local-search ambiguity: local search helps both QAOA and
random candidates, but candidate-aware QAOA already places the optimum inside
the raw top-8 set on the evaluated suite.

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
| No measured hardware result yet | Resolved. Two IBM `ibm_quebec` jobs completed: the 24-qubit headline bridge and the smaller 18-qubit bridge. Both are hardware-execution evidence, not hardware-advantage evidence. |
| Hardware path differs from headline valid-subspace solver | Still true. The hardware path remains a full-binary QUBO bridge and should not be claimed as the custom valid-subspace mixer. |
| Hardware projection may look too strong | Quantified. The 24-qubit job produced 16/1024 projected optimum count versus 73.7/1024 random projection baseline; the 18-qubit job produced 203/1024 versus 199.2/1024 random projection baseline. |
| Small scale | Reduced, not eliminated. The benchmark now includes a `U=4, G=7, S=7` scale challenge with 724 feasible assignments and 28 full-binary variables. |
| Local search may dominate | Reduced. The training notebook and README report raw top-K quality separately; candidate-aware QAOA reaches raw top-8 AR 1.000 on the scale case and multi-seed suite. |
| Simulated annealing is close | Kept explicit. SA remains a strong baseline, but trails candidate-aware QAOA + local in the recorded mean AR and optimum-hit metrics. |
| Noise robustness missing | Reduced with a probability-blend robustness probe and real hardware samples. The hardware results show raw feasibility remains poor, so projection/post-processing must be presented honestly. |
| `QAOA_ISAC_UAV.ipynb` correctness red flags | Quarantined as reference-only and ignored by git; it is not used as benchmark evidence. |

## Visual Evidence

The clearest current figures are in `qaoa_isac_training_notebook.ipynb`:

- Cell 11: standalone QAOA training convergence.
- Cell 15: system geometry, angle landscape, and candidate efficiency.
- Cell 17: multi-seed competition evidence.

The benchmark harness also generates presentation-ready PNGs in `figures/`.

![QAOA convergence](figures/qaoa_convergence.png)

![QAOA angle landscape](figures/qaoa_angle_landscape.png)

![Candidate efficiency](figures/qaoa_candidate_efficiency.png)

![Probability distribution](figures/qaoa_probability_distribution.png)

![Deployment map](figures/qaoa_deployment_map.png)

## Judging Criteria Mapping

| Criterion | Evidence in this repo |
| --- | --- |
| Innovation and originality | Constrained QAOA-guided UAV deployment for ISAC with SINR and collision constraints. |
| Technical depth | Exact feasible reference, greedy baseline, expected-QAOA and candidate-aware QAOA objectives, convergence traces, top-K sample-efficiency sweep, local-search attribution, simulated annealing baseline, and multi-seed stress suite. |
| Feasibility | Simulation-only training notebook, saved JSON evidence, reproducible benchmark harness, and guarded IBM Quantum Runtime hardware section. |
| Presentation | `qaoa_isac_training_notebook.ipynb` presents the training, convergence, multi-seed evidence, and export flow; `qaoa_isac_submission_evidence.json` gives the distilled final claims. |
| Teamwork | Add final named member contributions before submission once the team roster is confirmed. |

For the teamwork score, the final submission should name who owned the ISAC
model, QAOA solver, benchmark analysis, notebook/results, hardware path, and
presentation. Those names are not inferred in this README.

## Reproduce

Primary notebook workflow:

1. Open `qaoa_isac_training_notebook.ipynb`.
2. Restart the `qiskit` kernel.
3. Run all cells top to bottom.
4. Check Cell 17 for the multi-seed competition evidence.
5. Run Cell 21 to export `qaoa_isac_training_results.json`.

Optional harness regeneration:

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
| `qaoa_isac_training_notebook.ipynb` | Primary simulation-only QAOA training and multi-seed evidence notebook |
| `qaoa_isac_hardware_training_notebook.ipynb` | Separate opt-in notebook for real IBM hardware SPSA training of the 18-qubit full-binary QUBO bridge |
| `qaoa_isac_training_results.json` | Exported results from the training notebook |
| `qaoa_isac_submission_evidence.json` | Distilled final-submission evidence and safe claims |
| `docs/final-technical-presentation-brief.md` | Slide-ready technical explanation, speaker notes, and hardware-run checklist |
| `qaoa_isac_benchmark.py` | Reproducible benchmark harness |
| `qaoa_isac_benchmark.ipynb` | Judge-facing notebook view |
| `qaoa_isac_benchmark_results.json` | Regenerated benchmark and sweep metrics |
| `figures/` | Generated convergence, landscape, probability, efficiency, and deployment plots |
| `qaoa_isac_figures.pdf` | Existing figure artifact |

## Hardware Status

The notebook hardware path is guarded with `SUBMIT_TO_HARDWARE = False` to
avoid accidental resubmission. The hardware circuit is a full-binary QUBO
bridge, not the custom valid-subspace mixer used by the simulation headline.

Completed IBM evidence:

| Scenario | Backend | Job ID | Raw feasible samples | Projected optimum count | Random projection baseline | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `U=4, G=6, S=6`, 24 qubits | `ibm_quebec` | `d91i01vccmks73d56i80` | 1/1024 | 16/1024 | 73.7/1024 | Executed, but below random projection baseline |
| `U=3, G=6, S=5`, 18 qubits | `ibm_quebec` | `d91ttqmu9n7c73ane4jg` | 0/1024 | 203/1024 | 199.2/1024 | Smaller execution improved projection slightly over random |

The 18-qubit run used ISA depth 882 with 780 CZ gates. Projection recovered the
optimum assignment `[1, 0, 5]` with AR 1.000, but raw feasible sampling remained
0/1024. Present both IBM jobs as real-hardware executability evidence, not
hardware quantum-advantage evidence.

See `docs/ibm-hardware-run.md` for hardware-job details and the guarded
notebook switches.
