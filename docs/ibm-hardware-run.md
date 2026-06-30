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

## What The Limitation Means

The main winning result uses a valid-subspace QAOA simulator:

- It represents only valid UAV assignments as basis states.
- The mixer moves between valid assignments by relocating one UAV.
- This is why it can search the constrained deployment space efficiently.

The hardware section is different:

- It builds a full-binary QUBO circuit with one qubit per `UAV x grid` variable.
- For the hardware demo scenario, that is `U=3, G=6`, so `18` qubits.
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
