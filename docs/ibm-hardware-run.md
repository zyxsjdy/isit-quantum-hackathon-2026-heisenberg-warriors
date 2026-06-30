# IBM Quantum Hardware Run Guide

This project has a guarded IBM Quantum Runtime section in
`qaoa_isac_benchmark.ipynb`. Use it only after the simulated benchmark results
are finalized.

## Install Runtime Support

The current `qiskit` conda environment does not have `qiskit-ibm-runtime`
installed. Install it in the same environment used by the notebook:

```powershell
& 'C:\Users\harry\.conda\envs\qiskit\python.exe' -m pip install qiskit-ibm-runtime
```

Then restart the notebook kernel.

## Save IBM Quantum Credentials

Do not commit your token. Run this locally once, replacing the placeholder:

```powershell
& 'C:\Users\harry\.conda\envs\qiskit\python.exe' -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(token='YOUR_IBM_QUANTUM_TOKEN', overwrite=True)"
```

If your IBM account requires an instance or CRN, pass it as `instance='...'` in
the same call.

## Notebook Cells To Change

Open `qaoa_isac_benchmark.ipynb` and find the section:

```python
SUBMIT_TO_HARDWARE = False
IBM_INSTANCE = None
HARDWARE_SHOTS = 1024
```

Change only this when you are ready to submit:

```python
SUBMIT_TO_HARDWARE = True
```

If IBM gave you an instance string, also set:

```python
IBM_INSTANCE = "your-instance-or-crn"
```

Then run the hardware section cells in order. The code uses:

- `QiskitRuntimeService()`
- `service.least_busy(...)`
- `generate_preset_pass_manager(...)`
- `SamplerV2(mode=backend).run(..., shots=HARDWARE_SHOTS)`

When the job is submitted, save the printed job ID in the README or final
presentation notes.

## Hardware Evidence To Capture

Do not update the headline claim until the real job returns. When it does,
capture the following fields and compare every method on the same hardware
scenario:

| Field | Why it matters |
| --- | --- |
| `backend_name` | Confirms the actual IBM device. |
| `job_id` | Allows the run to be audited. |
| `qubit_count` | Shows the hardware scenario size. |
| `depth` | Reports compiled circuit depth. |
| `cx_count` | Reports two-qubit gate cost. |
| `shots` | Defines sampling budget. |
| `feasible_sample_rate` | Measures how much QUBO sampling survives constraints. |
| `best_feasible_bitstring` | Preserves the raw measured candidate. |
| `best_feasible_assignment` | Converts the bitstring into UAV placement. |
| `best_hardware_AR_rate` | Compares hardware quality against exact feasible optimum. |
| `same_scenario_greedy_AR_rate` | Keeps the hardware comparison fair. |
| `same_scenario_random_AR_rate` | Shows whether hardware beats random feasible sampling. |

The regenerated benchmark JSON has the same required field list under
`hardware_evidence`.

## Captured Hardware Result

The current completed job is:

| Field | Value |
| --- | --- |
| Backend | `ibm_quebec` |
| Job ID | `d91i01vccmks73d56i80` |
| Qubits | 24 |
| Shots | 1024 |
| Count register | `c` |
| ISA depth | 1233 |
| ISA two-qubit gates | 1572 `cz` gates |
| Raw feasible samples | 1/1024 |
| Best raw feasible AR | 0.829 |
| Best projected assignment | `[2, 4, 1, 3]` |
| Best projected AR | 1.000 |

Interpretation: the device executed the full-binary QUBO circuit, but raw
feasibility was very low. The optimum recovery came after feasible projection,
so this is hardware feasibility evidence, not a hardware quantum-advantage
claim.

## Next Smaller Hardware Target

The first hardware-improvement target is a smaller full-binary QUBO bridge:

| Field | Value |
| --- | --- |
| Scenario | `U=3, G=6, S=5` |
| Seed | `35` |
| Full-binary qubits | 18 |
| Feasible assignments | 22 |
| Greedy AR | 0.785 |
| Valid-subspace QAOA top AR | 1.000 |
| Valid-subspace QAOA optimum probability | 0.358 |
| Shots for 95% optimum sample in simulator | 7 |
| Pre-ISA circuit depth | 103 |
| Pre-ISA two-qubit gates | 306 `cx` gates |

This is not measured hardware evidence yet. It is the next candidate to submit
because it reduces the previous 24-qubit bridge from depth 139 and 552 CX gates
before ISA transpilation to depth 103 and 306 CX gates. Before submitting, run
the notebook hardware cells with this scenario and check the selected backend's
ISA depth and two-qubit gate count.

Regenerate the candidate evidence with:

```powershell
& 'C:\Users\harry\.conda\envs\qiskit\python.exe' -X utf8 .\qaoa_isac_benchmark.py --include-hardware-demo-candidate
```

## What The Limitation Means

The main winning result uses a valid-subspace QAOA simulator:

- It represents only valid UAV assignments as basis states.
- The mixer moves between valid assignments by relocating one UAV.
- This is why it can search the constrained deployment space efficiently.

The hardware section is different:

- It builds a full-binary QUBO circuit with one qubit per `UAV x grid` variable.
- The captured job used the headline `U=4, G=6` scenario, so `24` qubits.
- It does not implement the custom valid-subspace mixer used by the simulator.
- Constraint violations are handled by QUBO penalties and post-processing, not
  by staying in the valid assignment subspace.

So the safe presentation wording is:

> The headline optimization evidence comes from a valid-subspace QAOA simulator.
> The IBM hardware section demonstrates a hardware-executable full-binary QUBO
> path for the same ISAC deployment formulation.

Do not say the hardware job directly reproduces the custom valid-subspace QAOA
mixer unless that circuit synthesis is implemented later.

## References

- IBM Quantum: install Qiskit locally:
  https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit
- IBM Quantum: get started with Runtime primitives and `SamplerV2`:
  https://quantum.cloud.ibm.com/docs/en/guides/get-started-with-primitives
