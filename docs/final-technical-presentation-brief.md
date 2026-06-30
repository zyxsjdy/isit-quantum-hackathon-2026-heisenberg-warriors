# Final Technical Presentation Brief

Project: QAOA-ISAC UAV Deployment

Use this document as slide content plus speaker notes. The safe headline is:

> Candidate-aware valid-subspace QAOA is a reliable feasible-candidate
> generator for constrained ISAC UAV deployment. It trains QAOA angles in
> simulation, ranks feasible UAV assignments by probability, and polishes only
> the top-K candidates with local search.

Do not claim hardware quantum advantage. The IBM hardware runs are real-device
execution evidence for a full-binary QUBO bridge.

## Slide 1 - Problem

### On-Slide Message

UAV placement for integrated sensing and communication is a constrained
optimization problem.

- Choose where each UAV should fly.
- Serve ground survivors with strong communication links.
- Avoid UAV co-location and unsafe UAV separation.
- Respect minimum survivor SINR.

### Speaker Notes

The problem is not just placing UAVs near users. UAVs interfere with each other,
survivors need minimum SINR, and each UAV must occupy one candidate 3-D grid
point. The search space grows combinatorially as the number of UAVs and grid
points increases.

## Slide 2 - System Model

### On-Slide Message

Scenario:

- `U` UAVs.
- `G` candidate 3-D grid points.
- `S` ground survivors.
- Each UAV has `Nt` antennas.
- Binary decision variable: `x[u,g] = 1` when UAV `u` uses grid point `g`.

### Speaker Notes

The grid points are 3-D UAV candidate locations. Survivors are on the ground.
The model uses a physical air-to-ground communication channel: distance,
elevation angle, LoS probability, path loss, Rician channel, MRT beamforming,
SINR, and Shannon rate.

The final objective is the exact sum-rate:

```text
R_sum(x) = sum_s B log2(1 + SINR_s(x))
```

The approximation ratio reported in results is:

```text
AR = method_sum_rate / exact_optimum_sum_rate
```

## Slide 3 - Constraints

### On-Slide Message

Feasible deployment constraints:

| Constraint | Meaning |
| --- | --- |
| C1 | Each UAV chooses exactly one grid point. |
| C2 | No two UAVs occupy the same grid point. |
| C3 | Each survivor meets minimum SINR. |
| C4 | UAVs avoid unsafe separation/collision pairs. |

### Speaker Notes

C1 and C2 enforce assignment validity. C3 enforces communication quality. C4
prevents unsafe geometric placement. These constraints make the problem harder
than unconstrained bitstring search because most full-binary samples are not
valid deployments.

## Slide 4 - QUBO Formulation

### On-Slide Message

Full-binary QUBO uses one bit per UAV-grid pair:

```text
n = U * G
x in {0,1}^{U*G}
```

The QUBO minimizes:

```text
H_C(x) = -R_qubo(x)
       + lambda1 P1(x)
       + lambda2 P2(x)
       + lambda3 P3(x)
       + lambda4 P4(x)
```

### Speaker Notes

The sum-rate is maximized, so it enters the minimization Hamiltonian with a
negative sign. Constraint violations are added as penalties.

The code builds:

- a linearized rate approximation for QUBO construction,
- penalty terms for C1-C4,
- a QUBO matrix `Q`,
- and an Ising form `(J, h, c0)` for hardware-executable QAOA circuits.

The exact rate is still used for final scoring and validation.

## Slide 5 - Why Valid-Subspace QAOA

### On-Slide Message

Full-binary hardware samples many invalid bitstrings.

Valid-subspace QAOA instead searches only feasible UAV assignments:

```text
Basis state = one feasible assignment
```

### Speaker Notes

The key innovation is to separate the strongest algorithmic claim from the
hardware bridge. The valid-subspace solver enumerates feasible assignments and
uses them as the QAOA basis. This means QAOA probability mass is never wasted on
invalid one-hot or collision-violating samples.

This is why the simulator result is much cleaner than the IBM full-binary
hardware result.

## Slide 6 - Our QAOA Structure

### On-Slide Message

Valid-subspace p=1 QAOA:

```text
|psi0> = uniform superposition over feasible assignments
U_C(gamma) = exp(-i gamma C)
U_M(beta) = exp(-i beta A)
|psi(gamma,beta)> = U_M(beta) U_C(gamma) |psi0>
```

Where:

- `C` is the normalized negative exact rate.
- `A` is the assignment-graph adjacency matrix.
- Edges connect assignments that differ by one UAV relocation.

### Speaker Notes

This is not the same as the full-binary RX mixer. The valid-subspace mixer is
defined on the graph of feasible deployments. Two states are neighbors if they
only move one UAV. That gives the algorithm a physically meaningful local move.

In this repo the QAOA depth is p=1. The angle search is a grid search over:

```text
gamma in [0, 2*pi]
beta in [0, pi]
```

The notebook shows convergence as best-so-far AR over angle evaluations.

## Slide 7 - Candidate-Aware Training Objective

### On-Slide Message

Standard expected-rate QAOA optimizes:

```text
maximize E[R]
```

Our candidate-aware objective optimizes the actual downstream use:

```text
maximize best raw rate inside top-K probability-ranked candidates
```

### Speaker Notes

The method is used as a candidate generator. We do not need the single most
probable state to be the optimum. We need the optimum or a near-optimum state to
appear inside a small top-K set, because only that set is polished by local
search.

This change is important in the scale case:

- Expected-QAOA raw top-8 AR: 0.927.
- Candidate-aware raw top-8 AR: 1.000.
- Expected optimum probability: 0.117%.
- Candidate-aware optimum probability: 0.405%.

## Slide 8 - Local Search Role

### On-Slide Message

Pipeline:

```text
QAOA probabilities -> top-K assignments -> local search -> final deployment
```

Local search is not hidden:

- Raw top-K AR is reported.
- Local-search gain is reported.
- Random top-K + local is reported.

### Speaker Notes

This is important for judging. We are not pretending that local search does
nothing. We explicitly measure what QAOA contributes before local polishing.
Candidate-aware QAOA is strong because it already places the optimum inside the
raw top-K set on the scale case and multi-seed suite.

## Slide 9 - Main Simulation Evidence

### On-Slide Message

Headline case, `U=4, G=6, S=6`:

| Method | AR |
| --- | ---: |
| Greedy | 0.811 |
| Expected-QAOA raw top-8 | 1.000 |
| Candidate-aware QAOA top-8 + local | 1.000 |
| Simulated annealing mean | 0.993 |

### Speaker Notes

The headline case proves the full workflow on a nontrivial exact-validatable
scenario with 329 feasible assignments.

## Slide 10 - Scale Challenge

### On-Slide Message

Scale case, `U=4, G=7, S=7`:

| Method | AR |
| --- | ---: |
| Greedy + local | 0.835 |
| Expected-QAOA raw top-8 | 0.927 |
| Candidate-aware QAOA raw top-8 | 1.000 |
| Candidate-aware QAOA top-8 + local | 1.000 |
| Simulated annealing mean | 0.968 |

### Speaker Notes

This is the best single-case evidence for the improved method. Candidate-aware
training changes what the raw top-K candidates contain. It finds the optimum
before local polishing, while the expected-rate objective does not.

## Slide 11 - Multi-Seed Evidence

### On-Slide Message

Multi-seed suite, `U=3, G=6, S=5`, seeds `1:60`:

| Metric | Result |
| --- | ---: |
| Feasible evaluated cases | 25 |
| Greedy + local mean AR | 0.912 |
| Random top-8 + local mean AR | 0.990 |
| Simulated annealing mean AR | 0.986 |
| Candidate-aware raw top-8 mean AR | 1.000 |
| Candidate-aware top-8 + local mean AR | 1.000 |
| Candidate-aware optimum hits | 25/25 |

### Speaker Notes

This is the strongest robustness evidence. It says the method is not tuned to
one geometry. On every feasible evaluated case, the optimum appears in the
candidate-aware top-8 set.

## Slide 12 - Stress Suite

### On-Slide Message

Stress cases are seeds where greedy is at least 10% below optimum.

| Metric | Result |
| --- | ---: |
| Stress cases | 12 |
| Greedy + local mean AR | 0.850 |
| Random top-8 + local mean AR | 0.990 |
| Simulated annealing mean AR | 0.986 |
| Candidate-aware top-8 + local mean AR | 1.000 |
| Candidate-aware optimum hits | 12/12 |

### Speaker Notes

This is the best answer to "does the method help when greedy is weak?" Yes:
candidate-aware QAOA recovers the optimum on all stress cases in the recorded
suite.

## Slide 13 - Hardware Path

### On-Slide Message

IBM hardware uses the full-binary QUBO bridge:

```text
one qubit per x[u,g]
Hadamards -> QUBO phase separator -> RX mixer -> measurement
```

This is hardware-executable, but it is not the custom valid-subspace mixer.

### Speaker Notes

The hardware path is intentionally presented separately. The simulator headline
uses the valid-subspace assignment graph mixer. The IBM device runs a
full-binary QUBO circuit because that is directly implementable with standard
QAOA gates.

## Slide 14 - Completed IBM Runs

### On-Slide Message

| Scenario | Qubits | Raw feasible | Projected optimum | Random projection baseline |
| --- | ---: | ---: | ---: | ---: |
| `U=4,G=6,S=6` | 24 | 1/1024 | 16/1024 | 73.7/1024 |
| `U=3,G=6,S=5` | 18 | 0/1024 | 203/1024 | 199.2/1024 |

### Speaker Notes

The 18-qubit IBM job was `d91ttqmu9n7c73ane4jg` on `ibm_quebec`. It projected
to the optimum slightly above the random projection baseline, but raw feasible
sampling was still zero. Therefore this is improved hardware execution
evidence, not hardware quantum advantage.

Safe wording:

> IBM hardware executed the full-binary QUBO bridge for the same ISAC
> deployment formulation. The strongest optimization evidence remains the
> valid-subspace QAOA simulator.

## Slide 15 - What To Run On Real Hardware Now

### On-Slide Message

Use `qaoa_isac_benchmark.ipynb`, section 7.

Run order:

1. Set `CHECK_HARDWARE_TRANSPILATION = True`.
2. Keep `SUBMIT_TO_HARDWARE = False`.
3. Run hardware cells through the transpilation check.
4. Review backend, ISA depth, and CZ count.
5. Only then set `SUBMIT_TO_HARDWARE = True`.
6. Run the submit cell once.
7. Save the printed job ID.
8. After `DONE`, set `SUBMIT_TO_HARDWARE = False`.
9. Run the result extraction cell.

### Speaker Notes

Do not blindly submit. The job should only be sent if the transpiled circuit is
not worse than the completed 18-qubit run:

- Previous 18-qubit ISA depth: 882.
- Previous 18-qubit CZ count: 780.

If the new backend gives much higher depth or CZ count, wait or choose another
time.

## Slide 16 - If We Want Hardware Training

### On-Slide Message

Hardware training is possible, but expensive:

```text
for each optimizer step:
  choose gamma,beta
  run IBM Sampler
  estimate QUBO energy from counts
  update gamma,beta
```

Recommended only if time allows.

### Speaker Notes

Training on hardware is not the same as running trained angles on hardware. It
requires many queued jobs. A fair hardware-training objective should use the
measured QUBO energy or feasibility-aware projected energy, not the known exact
optimum label. A practical minimum experiment would use SPSA or COBYLA for
10-20 hardware evaluations with low shots, followed by one high-shot final run.

This would be stronger as a "future work" or "stretch result" unless queue time
is very favorable.

## Slide 17 - Why This Can Win

### On-Slide Message

Strengths:

- Real ISAC physical channel model.
- Clear QUBO mapping and constraints.
- Custom valid-subspace QAOA mixer.
- Candidate-aware objective matches actual top-K use.
- Exact optimum references for every evaluated case.
- Greedy, random top-K, and simulated annealing baselines.
- Multi-seed and stress-seed evidence.
- Real IBM hardware execution evidence.

### Speaker Notes

The strongest story is not "we beat all classical algorithms on hardware." The
strongest story is a clean quantum optimization workflow with honest baselines,
defensible constraints, and real hardware execution.

## Slide 18 - Safe Final Claims

### On-Slide Message

Claim:

> Candidate-aware valid-subspace QAOA reliably generates high-quality feasible
> UAV deployment candidates for constrained ISAC.

Do not claim:

- Hardware quantum advantage.
- Hardware-trained QAOA, unless a new hardware-training loop is actually run.
- That the IBM circuit implements the custom valid-subspace mixer.

### Speaker Notes

This wording is defensible. It avoids overclaiming while still showing a strong
method and strong evidence.

## Real Hardware Run Checklist

Use this when you actually want to submit another IBM job.

### Step 1 - Open The Correct Notebook

Open:

```text
qaoa_isac_benchmark.ipynb
```

Go to:

```text
Section 7. Optional IBM Quantum Hardware Run
```

### Step 2 - Transpilation Check First

In the hardware config cell, use:

```python
SUBMIT_TO_HARDWARE = False
CHECK_HARDWARE_TRANSPILATION = True
HARDWARE_SHOTS = 1024
HARDWARE_JOB_ID = ""
```

Run the hardware cells through the ISA transpilation cell. Record:

- selected backend,
- ISA depth,
- CZ/two-qubit gate count.

Submit only if the result is acceptable compared with the completed 18-qubit
job:

```text
depth <= about 882
CZ count <= about 780
```

### Step 3 - Submit One Job

If the transpilation looks acceptable, change only:

```python
SUBMIT_TO_HARDWARE = True
CHECK_HARDWARE_TRANSPILATION = True
```

Run the submit cell once. Copy the job ID immediately.

### Step 4 - Retrieve Results

After the job status is `DONE`, set:

```python
SUBMIT_TO_HARDWARE = False
HARDWARE_JOB_ID = "<new job id>"
```

Run the result extraction cell. Capture:

- raw feasible samples,
- best raw feasible AR,
- best projected assignment,
- best projected AR,
- projected optimum count,
- random projection baseline,
- ratio against random projection.

## Strongest Remaining Winning Moves

1. Create a 3-minute video or demo script from this document.
2. Add final team member names and roles to the README.
3. Use screenshots from `qaoa_isac_training_notebook.ipynb` Cell 11 and Cell 17.
4. Include one slide that says exactly what the IBM hardware run proves and
   what it does not prove.
5. If time and queue allow, run one more 18-qubit IBM job only after a good
   transpilation check. Treat it as additional hardware evidence, not the
   headline.
6. Stretch only if there is time: implement hardware-training with a small
   SPSA loop over `gamma,beta`, using measured QUBO energy as the objective.

## One-Minute Pitch

We solve constrained UAV placement for integrated sensing and communication.
The system model includes air-to-ground channel physics, SINR, rate, assignment
constraints, collision avoidance, and minimum survivor QoS. We formulate a QUBO
for hardware execution, but our strongest optimization method is a
valid-subspace QAOA solver that searches only feasible UAV assignments. The
phase separator uses exact deployment rate, and the mixer moves between
assignments by relocating one UAV.

The key improvement is candidate-aware QAOA. Instead of maximizing only
expected rate, we train angles to make the top-K probability-ranked candidates
useful, because those are the candidates polished by local search. On the
`U=4,G=7,S=7` scale case, candidate-aware QAOA improves raw top-8 AR from 0.927
to 1.000. Across the multi-seed suite, it places the optimum in the top-8 set
on 25/25 feasible cases and 12/12 stress cases. We also ran IBM hardware on
full-binary QUBO circuits, including an 18-qubit run on `ibm_quebec`. The
hardware results demonstrate real-device executability, while the headline
optimization evidence remains the valid-subspace QAOA simulator.
