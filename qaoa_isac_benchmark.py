"""Harder QAOA-ISAC benchmark over the valid assignment subspace.

Question being tested:
Can a constraint-preserving QAOA-style solver produce a stronger competition
result than the greedy baseline on a larger scenario where full binary Qiskit
QAOA is too slow?

This benchmark keeps the physical channel/rate model from qaoa_isac_env.py, but
restricts the quantum search space to feasible UAV-to-grid assignments. The
phase separator uses the exact sum-rate objective on that finite valid basis,
and the mixer connects assignments that differ by one UAV relocation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from qaoa_isac_env import (
    ISACEnvironment,
    SystemParams,
    compute_rate,
    compute_sinr,
)


@dataclass(frozen=True)
class AssignmentEval:
    assignment: tuple[int, ...]
    sum_rate: float
    sinr: tuple[float, ...]
    qubo_energy: float
    c1: bool
    c2: bool
    c3: bool
    c4: bool

    @property
    def feasible(self) -> bool:
        return self.c1 and self.c2 and self.c3 and self.c4

    def to_json(self, exact_rate: float | None = None) -> dict:
        payload = asdict(self)
        payload["assignment"] = list(self.assignment)
        payload["sinr"] = list(self.sinr)
        payload["feasible"] = self.feasible
        if exact_rate:
            payload["AR_rate"] = self.sum_rate / exact_rate
        return payload


@dataclass(frozen=True)
class QAOASubspaceResult:
    reps: int
    grid_steps: int
    evaluations: int
    gamma: float
    beta: float
    expected_rate: float
    top_index: int
    top_probability: float
    optimum_probability: float
    sampled_best_index: int
    sampled_best_count: int
    shots: int
    best_so_far_trace: tuple[dict, ...] = ()
    angle_landscape: tuple[tuple[float, ...], ...] = ()
    gamma_grid: tuple[float, ...] = ()
    beta_grid: tuple[float, ...] = ()


@dataclass(frozen=True)
class LocalSearchComparison:
    top_k: int
    qaoa_raw_best: AssignmentEval
    qaoa_best: AssignmentEval
    qaoa_local_gain_ar: float
    random_raw_mean_ar: float
    random_raw_std_ar: float
    random_raw_worst_ar: float
    random_raw_best_ar: float
    random_local_gain_mean_ar: float
    random_mean_ar: float
    random_std_ar: float
    random_worst_ar: float
    random_best_ar: float
    random_optimum_hit_rate: float
    qaoa_beats_random_trial_rate: float
    random_trials: int

    def to_json(self, exact_rate: float) -> dict:
        return {
            "top_k": self.top_k,
            "qaoa_raw_best": self.qaoa_raw_best.to_json(exact_rate),
            "qaoa_best": self.qaoa_best.to_json(exact_rate),
            "qaoa_local_gain_AR_rate": self.qaoa_local_gain_ar,
            "random_raw_mean_AR_rate": self.random_raw_mean_ar,
            "random_raw_std_AR_rate": self.random_raw_std_ar,
            "random_raw_worst_AR_rate": self.random_raw_worst_ar,
            "random_raw_best_AR_rate": self.random_raw_best_ar,
            "random_local_gain_mean_AR_rate": self.random_local_gain_mean_ar,
            "random_mean_AR_rate": self.random_mean_ar,
            "random_std_AR_rate": self.random_std_ar,
            "random_worst_AR_rate": self.random_worst_ar,
            "random_best_AR_rate": self.random_best_ar,
            "random_optimum_hit_rate": self.random_optimum_hit_rate,
            "qaoa_beats_random_trial_rate": self.qaoa_beats_random_trial_rate,
            "random_trials": self.random_trials,
        }


@dataclass(frozen=True)
class SimulatedAnnealingComparison:
    restarts: int
    steps_per_restart: int
    trials: int
    start_temperature: float
    end_temperature: float
    best: AssignmentEval
    mean_ar: float
    std_ar: float
    worst_ar: float
    best_ar: float
    optimum_hit_rate: float

    def to_json(self, exact_rate: float) -> dict:
        return {
            "restarts": self.restarts,
            "steps_per_restart": self.steps_per_restart,
            "trials": self.trials,
            "start_temperature": self.start_temperature,
            "end_temperature": self.end_temperature,
            "best": self.best.to_json(exact_rate),
            "mean_AR_rate": self.mean_ar,
            "std_AR_rate": self.std_ar,
            "worst_AR_rate": self.worst_ar,
            "best_AR_rate": self.best_ar,
            "optimum_hit_rate": self.optimum_hit_rate,
        }


def build_environment(params: SystemParams, seed: int, quiet: bool = True) -> ISACEnvironment:
    if not quiet:
        return ISACEnvironment(params, seed=seed)

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            return ISACEnvironment(params, seed=seed)


def assignment_matrix(assignment: Sequence[int], u_count: int, g_count: int) -> np.ndarray:
    x = np.zeros((u_count, g_count), dtype=int)
    for u, g in enumerate(assignment):
        x[u, g] = 1
    return x


def exact_rate_for_matrix(env: ISACEnvironment, x: np.ndarray) -> tuple[float, tuple[float, ...]]:
    sinr = tuple(
        compute_sinr(x, env.G_table, env.I_table, env.params.sigma2, s)
        for s in range(env.params.S)
    )
    rate = sum(compute_rate(value, env.params.B) for value in sinr)
    return rate, sinr


def evaluate_assignment(env: ISACEnvironment, assignment: Sequence[int]) -> AssignmentEval:
    assignment_tuple = tuple(int(g) for g in assignment)
    u_count = env.params.U
    g_count = env.params.G
    c1 = len(assignment_tuple) == u_count
    c2 = len(set(assignment_tuple)) == len(assignment_tuple)

    exclusion = set(env.exclusion)
    c4 = True
    for u in range(u_count):
        for v in range(u + 1, u_count):
            pair = (assignment_tuple[u], assignment_tuple[v])
            if pair in exclusion or (pair[1], pair[0]) in exclusion:
                c4 = False

    x = assignment_matrix(assignment_tuple, u_count, g_count)
    rate, sinr = exact_rate_for_matrix(env, x)
    c3 = all(value >= env.params.Gamma_min for value in sinr)
    flat = x.reshape(-1)
    qubo_energy = float(flat @ env.Q @ flat)

    return AssignmentEval(
        assignment=assignment_tuple,
        sum_rate=float(rate),
        sinr=tuple(float(value) for value in sinr),
        qubo_energy=qubo_energy,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
    )


def bitstring_to_variable_vector(bitstring: str, n_qubits: int) -> np.ndarray:
    # Qiskit count strings are displayed with the highest classical bit first.
    return np.array([int(bit) for bit in bitstring[::-1][:n_qubits]], dtype=int)


def assignment_vector(row: AssignmentEval, env: ISACEnvironment) -> np.ndarray:
    x = np.zeros(env.n_qubits, dtype=int)
    for u, g in enumerate(row.assignment):
        x[u * env.params.G + g] = 1
    return x


def project_bitstring_to_feasible_assignment(
    bitstring: str,
    feasible_rows: Sequence[AssignmentEval],
    env: ISACEnvironment,
) -> AssignmentEval:
    bits = bitstring_to_variable_vector(bitstring, env.n_qubits)
    best_row = None
    best_key = None
    for row in feasible_rows:
        candidate = assignment_vector(row, env)
        hamming = int(np.sum(bits != candidate))
        key = (-hamming, row.sum_rate)
        if best_key is None or key > best_key:
            best_key = key
            best_row = row
    if best_row is None:
        raise ValueError("At least one feasible row is required.")
    return best_row


def summarize_hardware_counts(
    counts: dict[str, int],
    feasible_rows: Sequence[AssignmentEval],
    env: ISACEnvironment,
    exact_rate: float,
) -> list[dict]:
    total_shots = sum(counts.values())
    if total_shots <= 0:
        raise ValueError("Hardware counts must contain at least one shot.")

    repaired: Counter[tuple[int, ...]] = Counter()
    for bitstring, count in counts.items():
        row = project_bitstring_to_feasible_assignment(bitstring, feasible_rows, env)
        repaired[row.assignment] += int(count)

    summary = []
    lookup = {row.assignment: row for row in feasible_rows}
    for assignment, count in repaired.most_common(10):
        row = lookup[assignment]
        summary.append(
            {
                "assignment": list(assignment),
                "count": int(count),
                "probability": count / total_shots,
                "sum_rate_mbps": row.sum_rate / 1e6,
                "AR_rate": row.sum_rate / exact_rate,
            }
        )
    return summary


def _databin_field_names(data: Any) -> list[str]:
    names: list[str] = []
    if hasattr(data, "keys"):
        names.extend(str(name) for name in data.keys())
    if hasattr(data, "items"):
        names.extend(str(name) for name, _ in data.items())
    if hasattr(data, "_FIELDS"):
        names.extend(str(name) for name in data._FIELDS)
    if hasattr(data, "__dict__"):
        names.extend(name for name in data.__dict__ if not name.startswith("_"))

    seen: set[str] = set()
    unique_names: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    return unique_names


def extract_sampler_counts(
    sampler_result: Any,
    classical_register: str | None = None,
    *,
    return_register: bool = False,
) -> dict[str, int] | tuple[dict[str, int], str]:
    """Return counts from a Qiskit Runtime SamplerV2 result.

    SamplerV2 stores one BitArray per classical register in a DataBin. The
    register name depends on how the circuit was measured: `measure_all()`
    commonly creates `meas`, while `QuantumCircuit(n, n)` creates `c`.
    """
    pub_result = sampler_result[0] if hasattr(sampler_result, "__getitem__") else sampler_result
    data = getattr(pub_result, "data", pub_result)

    candidates: list[tuple[str, Any]] = []
    if classical_register is not None:
        candidates.append((classical_register, getattr(data, classical_register, None)))
    if hasattr(data, "get_counts"):
        candidates.append(("<direct>", data))
    for name in _databin_field_names(data):
        candidates.append((name, getattr(data, name, None)))
    for fallback_name in ("meas", "c"):
        candidates.append((fallback_name, getattr(data, fallback_name, None)))

    seen: set[str] = set()
    for name, value in candidates:
        if name in seen or value is None:
            continue
        seen.add(name)
        get_counts = getattr(value, "get_counts", None)
        if callable(get_counts):
            counts = {str(key): int(count) for key, count in get_counts().items()}
            if return_register:
                return counts, name
            return counts

    available = ", ".join(_databin_field_names(data)) or "<none>"
    raise AttributeError(
        "No classical register with get_counts() was found in the Sampler result. "
        f"Available DataBin fields: {available}"
    )


def scaled_ising_terms(env: ISACEnvironment) -> tuple[np.ndarray, np.ndarray, float]:
    h = np.array(env.h_bias, dtype=float)
    j = np.array(env.J, dtype=float)
    scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(j))), 1.0)
    return h / scale, j / scale, scale


def build_full_binary_qaoa_circuit(
    env: ISACEnvironment,
    gamma: float,
    beta: float,
    *,
    reps: int = 1,
    measure: bool = True,
    coupling_threshold: float = 0.0,
) -> Any:
    from qiskit import QuantumCircuit

    n_qubits = env.n_qubits
    circuit = QuantumCircuit(n_qubits, n_qubits if measure else 0)
    h, j, scale = scaled_ising_terms(env)

    circuit.h(range(n_qubits))
    for _ in range(reps):
        for i, coeff in enumerate(h):
            if abs(coeff) > 1e-12:
                circuit.rz(2.0 * gamma * coeff, i)
        for i in range(n_qubits):
            for k in range(i + 1, n_qubits):
                coeff = j[i, k]
                if abs(coeff) <= max(coupling_threshold, 1e-12):
                    continue
                circuit.cx(i, k)
                circuit.rz(2.0 * gamma * coeff, k)
                circuit.cx(i, k)
        for i in range(n_qubits):
            circuit.rx(2.0 * beta, i)

    if measure:
        circuit.measure(range(n_qubits), range(n_qubits))
    circuit.metadata = {
        "problem": "QAOA-ISAC full-binary QUBO",
        "U": env.params.U,
        "G": env.params.G,
        "S": env.params.S,
        "Gamma_min": env.params.Gamma_min,
        "qubo_scale": scale,
        "gamma": gamma,
        "beta": beta,
        "reps": reps,
        "coupling_threshold": coupling_threshold,
    }
    return circuit


def circuit_summary(circuit: Any) -> dict:
    ops = {str(name): int(count) for name, count in circuit.count_ops().items()}
    two_qubit_names = {"cx", "cz", "ecr", "swap", "rzz"}
    return {
        "num_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth()),
        "ops": ops,
        "two_qubit_gate_count": int(
            sum(count for name, count in ops.items() if name in two_qubit_names)
        ),
    }


def enumerate_assignments(env: ISACEnvironment, require_c3: bool) -> list[AssignmentEval]:
    rows: list[AssignmentEval] = []
    for assignment in permutations(range(env.params.G), env.params.U):
        row = evaluate_assignment(env, assignment)
        if not (row.c1 and row.c2 and row.c4):
            continue
        if require_c3 and not row.c3:
            continue
        rows.append(row)
    return rows


def partial_exact_rate(env: ISACEnvironment, partial_assignment: Sequence[int | None]) -> float:
    x = np.zeros((env.params.U, env.params.G), dtype=int)
    for u, g in enumerate(partial_assignment):
        if g is not None:
            x[u, int(g)] = 1
    rate, _ = exact_rate_for_matrix(env, x)
    return float(rate)


def partial_c4_ok(env: ISACEnvironment, partial_assignment: Sequence[int | None]) -> bool:
    exclusion = set(env.exclusion)
    assigned = [(u, int(g)) for u, g in enumerate(partial_assignment) if g is not None]
    for i, (_, g) in enumerate(assigned):
        for _, gp in assigned[i + 1 :]:
            if (g, gp) in exclusion or (gp, g) in exclusion:
                return False
    return True


def greedy_assignment(env: ISACEnvironment) -> AssignmentEval:
    assignment: list[int | None] = [None] * env.params.U
    unused = set(range(env.params.G))

    for u in range(env.params.U):
        best_grid = None
        best_rate = -math.inf
        for g in sorted(unused):
            candidate = assignment[:]
            candidate[u] = g
            if not partial_c4_ok(env, candidate):
                continue
            rate = partial_exact_rate(env, candidate)
            if rate > best_rate:
                best_rate = rate
                best_grid = g
        if best_grid is None:
            raise RuntimeError("Greedy could not find a collision-safe grid point.")
        assignment[u] = best_grid
        unused.remove(best_grid)

    return evaluate_assignment(env, [int(g) for g in assignment])


def local_search(env: ISACEnvironment, start: AssignmentEval) -> AssignmentEval:
    current = start
    improved = True
    while improved:
        improved = False
        used = set(current.assignment)
        for u in range(env.params.U):
            for g in range(env.params.G):
                if g in used and g != current.assignment[u]:
                    continue
                candidate = list(current.assignment)
                candidate[u] = g
                if len(set(candidate)) != len(candidate):
                    continue
                row = evaluate_assignment(env, candidate)
                if not (row.c1 and row.c2 and row.c3 and row.c4):
                    continue
                if row.sum_rate > current.sum_rate + 1e-9:
                    current = row
                    improved = True
                    break
            if improved:
                break
    return current


def best_local_search_from_candidates(
    env: ISACEnvironment,
    candidates: Sequence[AssignmentEval],
) -> AssignmentEval:
    best: AssignmentEval | None = None
    for candidate in candidates:
        polished = local_search(env, candidate)
        if best is None or polished.sum_rate > best.sum_rate:
            best = polished
    if best is None:
        raise ValueError("At least one local-search candidate is required.")
    return best


def build_assignment_adjacency(states: Sequence[AssignmentEval]) -> np.ndarray:
    n_states = len(states)
    adjacency = np.zeros((n_states, n_states), dtype=float)
    for i, left in enumerate(states):
        for j in range(i + 1, n_states):
            right = states[j]
            distance = sum(a != b for a, b in zip(left.assignment, right.assignment))
            if distance == 1:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
    return adjacency


def run_valid_subspace_qaoa(
    states: Sequence[AssignmentEval],
    exact_best_index: int,
    *,
    reps: int = 1,
    grid_steps: int = 61,
    shots: int = 1024,
    seed: int = 35,
    record_diagnostics: bool = False,
) -> tuple[QAOASubspaceResult, np.ndarray]:
    if reps != 1:
        raise ValueError("This benchmark currently implements the p=1 subspace solver.")
    if len(states) < 2:
        raise ValueError("At least two feasible states are required for QAOA.")

    rates = np.array([row.sum_rate for row in states], dtype=float)
    exact_rate = float(states[exact_best_index].sum_rate)
    rate_span = float(rates.max() - rates.min())
    if rate_span <= 0:
        raise ValueError("All feasible assignments have identical rates.")

    normalized_cost = -((rates - rates.min()) / rate_span)
    adjacency = build_assignment_adjacency(states)
    mixer_evals, mixer_evecs = np.linalg.eigh(adjacency)
    initial_state = np.ones(len(states), dtype=complex) / math.sqrt(len(states))

    def apply_mixer(psi: np.ndarray, beta: float) -> np.ndarray:
        spectral = mixer_evecs.conj().T @ psi
        spectral *= np.exp(-1j * beta * mixer_evals)
        return mixer_evecs @ spectral

    def evaluate_angles(gamma: float, beta: float) -> tuple[float, np.ndarray]:
        psi = np.exp(-1j * gamma * normalized_cost) * initial_state
        psi = apply_mixer(psi, beta)
        probabilities = np.abs(psi) ** 2
        expected_rate = float(probabilities @ rates)
        return expected_rate, probabilities

    best_key: tuple[float, float, float] | None = None
    best_gamma = 0.0
    best_beta = 0.0
    best_expected = -math.inf
    best_probabilities = np.full(len(states), 1.0 / len(states))
    best_trace: list[dict] = []
    angle_landscape = (
        np.empty((grid_steps, grid_steps), dtype=float) if record_diagnostics else None
    )

    gamma_values = np.linspace(0.0, 2.0 * math.pi, grid_steps)
    beta_values = np.linspace(0.0, math.pi, grid_steps)
    evaluations = 0
    for gamma_index, gamma in enumerate(gamma_values):
        for beta_index, beta in enumerate(beta_values):
            evaluations += 1
            expected_rate, probabilities = evaluate_angles(float(gamma), float(beta))
            top_index = int(np.argmax(probabilities))
            expected_ar = expected_rate / exact_rate
            if angle_landscape is not None:
                angle_landscape[gamma_index, beta_index] = expected_ar
            key = (
                expected_rate,
                states[top_index].sum_rate,
                float(probabilities[top_index]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_gamma = float(gamma)
                best_beta = float(beta)
                best_expected = expected_rate
                best_probabilities = probabilities
                if record_diagnostics:
                    best_trace.append(
                        {
                            "evaluation": evaluations,
                            "gamma": float(gamma),
                            "beta": float(beta),
                            "best_expected_AR_rate": expected_ar,
                            "top_AR_rate": states[top_index].sum_rate / exact_rate,
                            "top_probability": float(probabilities[top_index]),
                            "optimum_probability": float(
                                probabilities[exact_best_index]
                            ),
                        }
                    )

    rng = np.random.default_rng(seed)
    sample_counts = rng.multinomial(shots, best_probabilities)
    sampled_indices = np.flatnonzero(sample_counts)
    sampled_best_index = int(max(sampled_indices, key=lambda i: states[int(i)].sum_rate))
    top_index = int(np.argmax(best_probabilities))

    result = QAOASubspaceResult(
        reps=reps,
        grid_steps=grid_steps,
        evaluations=evaluations,
        gamma=best_gamma,
        beta=best_beta,
        expected_rate=float(best_expected),
        top_index=top_index,
        top_probability=float(best_probabilities[top_index]),
        optimum_probability=float(best_probabilities[exact_best_index]),
        sampled_best_index=sampled_best_index,
        sampled_best_count=int(sample_counts[sampled_best_index]),
        shots=shots,
        best_so_far_trace=tuple(best_trace),
        angle_landscape=(
            tuple(tuple(float(value) for value in row) for row in angle_landscape)
            if angle_landscape is not None
            else ()
        ),
        gamma_grid=tuple(float(value) for value in gamma_values)
        if record_diagnostics
        else (),
        beta_grid=tuple(float(value) for value in beta_values)
        if record_diagnostics
        else (),
    )
    return result, best_probabilities


def summarize_random_baseline(states: Sequence[AssignmentEval]) -> dict:
    rates = np.array([row.sum_rate for row in states], dtype=float)
    best = float(rates.max())
    return {
        "mean_rate": float(rates.mean()),
        "std_rate": float(rates.std(ddof=0)),
        "best_rate": best,
        "uniform_optimum_probability": 1.0 / len(states),
    }


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",")]
        parsed = [int(part) for part in raw_values if part]
    else:
        parsed = [int(part) for part in value]

    seen: set[int] = set()
    result: list[int] = []
    for item in parsed:
        if item <= 0:
            raise ValueError("Top-k sweep values must be positive integers.")
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def parse_float_list(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",")]
        parsed = [float(part) for part in raw_values if part]
    else:
        parsed = [float(part) for part in value]

    seen: set[float] = set()
    result: list[float] = []
    for item in parsed:
        if item < 0.0 or item > 1.0:
            raise ValueError("Probability-noise levels must be in [0, 1].")
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def random_top_k_exact_hit_probability(
    states: Sequence[AssignmentEval],
    exact_rate: float,
    top_k: int,
) -> float:
    population = len(states)
    draws = min(top_k, population)
    optimal_count = sum(row.sum_rate >= exact_rate - 1e-9 for row in states)
    if optimal_count <= 0:
        return 0.0
    if draws >= population:
        return 1.0
    misses = math.comb(population - optimal_count, draws) / math.comb(population, draws)
    return 1.0 - misses


def compare_qaoa_vs_random_local_search(
    env: ISACEnvironment,
    states: Sequence[AssignmentEval],
    probabilities: np.ndarray,
    exact: AssignmentEval,
    *,
    top_k: int,
    random_trials: int,
    seed: int,
) -> LocalSearchComparison:
    if random_trials <= 0:
        raise ValueError("random_trials must be positive.")

    ordered = np.argsort(-probabilities)
    qaoa_candidates = [states[int(i)] for i in ordered[: min(top_k, len(states))]]
    qaoa_raw_best = max(qaoa_candidates, key=lambda row: row.sum_rate)
    qaoa_raw_ar = qaoa_raw_best.sum_rate / exact.sum_rate
    qaoa_best = best_local_search_from_candidates(env, qaoa_candidates)
    qaoa_ar = qaoa_best.sum_rate / exact.sum_rate

    rng = np.random.default_rng(seed + 2026)
    random_raw_ars: list[float] = []
    random_ars: list[float] = []
    random_local_gains: list[float] = []
    random_hits = 0
    random_losses = 0
    for _ in range(random_trials):
        size = min(top_k, len(states))
        random_indices = rng.choice(len(states), size=size, replace=False)
        random_candidates = [states[int(i)] for i in random_indices]
        random_raw_best = max(random_candidates, key=lambda row: row.sum_rate)
        random_best = best_local_search_from_candidates(env, random_candidates)
        raw_ar_rate = random_raw_best.sum_rate / exact.sum_rate
        ar_rate = random_best.sum_rate / exact.sum_rate
        random_raw_ars.append(raw_ar_rate)
        random_ars.append(ar_rate)
        random_local_gains.append(ar_rate - raw_ar_rate)
        if ar_rate >= 1.0 - 1e-9:
            random_hits += 1
        if qaoa_ar > ar_rate + 1e-9:
            random_losses += 1

    random_raw_array = np.array(random_raw_ars, dtype=float)
    random_array = np.array(random_ars, dtype=float)
    random_gain_array = np.array(random_local_gains, dtype=float)

    return LocalSearchComparison(
        top_k=top_k,
        qaoa_raw_best=qaoa_raw_best,
        qaoa_best=qaoa_best,
        qaoa_local_gain_ar=qaoa_ar - qaoa_raw_ar,
        random_raw_mean_ar=float(random_raw_array.mean()),
        random_raw_std_ar=float(random_raw_array.std(ddof=0)),
        random_raw_worst_ar=float(random_raw_array.min()),
        random_raw_best_ar=float(random_raw_array.max()),
        random_local_gain_mean_ar=float(random_gain_array.mean()),
        random_mean_ar=float(random_array.mean()),
        random_std_ar=float(random_array.std(ddof=0)),
        random_worst_ar=float(random_array.min()),
        random_best_ar=float(random_array.max()),
        random_optimum_hit_rate=random_hits / random_trials,
        qaoa_beats_random_trial_rate=random_losses / random_trials,
        random_trials=random_trials,
    )


def build_top_k_sweep(
    env: ISACEnvironment,
    states: Sequence[AssignmentEval],
    probabilities: np.ndarray,
    exact: AssignmentEval,
    *,
    top_k_values: Sequence[int],
    random_trials: int,
    seed: int,
) -> list[dict]:
    ordered = np.argsort(-probabilities)
    rows: list[dict] = []
    for requested_top_k in parse_int_list(top_k_values):
        effective_top_k = min(requested_top_k, len(states))
        comparison = compare_qaoa_vs_random_local_search(
            env,
            states,
            probabilities,
            exact,
            top_k=effective_top_k,
            random_trials=random_trials,
            seed=seed + requested_top_k * 7919,
        )
        qaoa_ar = comparison.qaoa_best.sum_rate / exact.sum_rate
        qaoa_top_indices = ordered[:effective_top_k]
        rows.append(
            {
                "top_k": requested_top_k,
                "effective_top_k": effective_top_k,
                "qaoa_raw_AR_rate": comparison.qaoa_raw_best.sum_rate / exact.sum_rate,
                "qaoa_AR_rate": qaoa_ar,
                "qaoa_local_gain_AR_rate": comparison.qaoa_local_gain_ar,
                "qaoa_optimum_hit": qaoa_ar >= 1.0 - 1e-9,
                "qaoa_assignment": list(comparison.qaoa_best.assignment),
                "qaoa_probability_mass": float(probabilities[qaoa_top_indices].sum()),
                "random_raw_mean_AR_rate": comparison.random_raw_mean_ar,
                "random_raw_std_AR_rate": comparison.random_raw_std_ar,
                "random_raw_worst_AR_rate": comparison.random_raw_worst_ar,
                "random_raw_best_AR_rate": comparison.random_raw_best_ar,
                "random_local_gain_mean_AR_rate": comparison.random_local_gain_mean_ar,
                "random_mean_AR_rate": comparison.random_mean_ar,
                "random_std_AR_rate": comparison.random_std_ar,
                "random_worst_AR_rate": comparison.random_worst_ar,
                "random_best_AR_rate": comparison.random_best_ar,
                "random_optimum_hit_rate": comparison.random_optimum_hit_rate,
                "random_exact_candidate_hit_probability": (
                    random_top_k_exact_hit_probability(states, exact.sum_rate, effective_top_k)
                ),
                "qaoa_gain_over_random_mean": qaoa_ar - comparison.random_mean_ar,
                "qaoa_beats_random_trial_rate": comparison.qaoa_beats_random_trial_rate,
                "random_trials": random_trials,
            }
        )
    return rows


def build_probability_noise_robustness(
    env: ISACEnvironment,
    states: Sequence[AssignmentEval],
    probabilities: np.ndarray,
    exact: AssignmentEval,
    *,
    top_k: int,
    shots: int,
    noise_levels: Sequence[float],
) -> dict:
    uniform = np.full(len(states), 1.0 / len(states), dtype=float)
    optimum_indices = [
        index
        for index, row in enumerate(states)
        if row.sum_rate >= exact.sum_rate - 1e-9
    ]
    rows: list[dict] = []

    for noise_level in parse_float_list(noise_levels):
        noisy_probabilities = (1.0 - noise_level) * probabilities + noise_level * uniform
        ordered = np.argsort(-noisy_probabilities)
        top_index = int(ordered[0])
        effective_top_k = min(top_k, len(states))
        top_k_indices = ordered[:effective_top_k]
        top_k_candidates = [states[int(i)] for i in top_k_indices]
        top_k_best = best_local_search_from_candidates(env, top_k_candidates)
        optimum_probability = float(noisy_probabilities[optimum_indices].sum())

        rows.append(
            {
                "uniform_blend": float(noise_level),
                "top_k": top_k,
                "effective_top_k": effective_top_k,
                "top_assignment": states[top_index].to_json(exact.sum_rate),
                "top_probability": float(noisy_probabilities[top_index]),
                "top_k_probability_mass": float(noisy_probabilities[top_k_indices].sum()),
                "top_k_local_assignment": top_k_best.to_json(exact.sum_rate),
                "top_k_local_AR_rate": top_k_best.sum_rate / exact.sum_rate,
                "top_k_optimum_hit": top_k_best.sum_rate >= exact.sum_rate - 1e-9,
                "optimum_probability": optimum_probability,
                "enrichment_vs_uniform": optimum_probability
                / random_top_k_exact_hit_probability(states, exact.sum_rate, 1),
                "shot_success_probability": 1.0
                - (1.0 - optimum_probability) ** shots,
                "shots_for_95pct_success": shots_for_success(optimum_probability),
            }
        )

    return {
        "model": "Convex blend of QAOA feasible-subspace probabilities toward uniform feasible sampling.",
        "interpretation": (
            "This is a simulator-side stress test of probability concentration, "
            "not a hardware noise calibration."
        ),
        "rows": rows,
    }


def build_hardware_evidence_status() -> dict:
    return {
        "status": "measured",
        "headline_source": "valid_subspace_qaoa_simulator",
        "hardware_path": "full_binary_qubo_bridge",
        "backend_name": "ibm_quebec",
        "job_id": "d91i01vccmks73d56i80",
        "count_register": "c",
        "qubit_count": 24,
        "pre_isa_depth": 139,
        "pre_isa_ops": {
            "cx": 552,
            "rz": 300,
            "h": 24,
            "rx": 24,
            "measure": 24,
        },
        "transpiled_depth": 1233,
        "transpiled_ops": {
            "sx": 2865,
            "rz": 1804,
            "cz": 1572,
            "measure": 24,
            "x": 1,
        },
        "cx_count": 0,
        "cz_count": 1572,
        "two_qubit_gate_count": 1572,
        "shots": 1024,
        "distinct_bitstrings": 1022,
        "raw_feasible_count": 1,
        "feasible_sample_rate": 1.0 / 1024.0,
        "best_feasible_bitstring": "001000000001100000000010",
        "best_feasible_assignment": [1, 5, 0, 3],
        "best_hardware_AR_rate": 0.8294928238727217,
        "best_projected_source_bitstring": "111001101011010010010100",
        "best_projected_assignment": [2, 4, 1, 3],
        "best_projected_AR_rate": 1.0,
        "best_projected_count": 16,
        "most_common_projected": [
            {
                "assignment": [4, 2, 3, 5],
                "count": 54,
                "AR_rate": 0.9074040406246278,
            },
            {
                "assignment": [0, 3, 1, 2],
                "count": 49,
                "AR_rate": 0.9114109085907078,
            },
            {
                "assignment": [0, 2, 3, 4],
                "count": 48,
                "AR_rate": 0.8869284583160361,
            },
            {
                "assignment": [2, 4, 3, 5],
                "count": 37,
                "AR_rate": 0.9921412175910209,
            },
            {
                "assignment": [5, 4, 3, 1],
                "count": 30,
                "AR_rate": 0.8874307484159896,
            },
        ],
        "same_scenario_greedy_AR_rate": 0.8111447260708325,
        "same_scenario_random_AR_rate": 0.7630645828543176,
        "same_scenario_qaoa_top_k_local_AR_rate": 1.0,
        "interpretation": (
            "Hardware executed the full-binary QUBO bridge on IBM Quantum. "
            "Raw feasible sampling was very low; feasible projection recovered "
            "the exact optimum, so this is hardware feasibility evidence rather "
            "than hardware quantum-advantage evidence."
        ),
    }


def run_hardware_demo_candidate(args: argparse.Namespace) -> dict:
    params = SystemParams(
        U=args.hardware_demo_uavs,
        G=args.hardware_demo_grid_points,
        S=args.hardware_demo_survivors,
        Nt=args.hardware_demo_antennas,
        Gamma_min=args.hardware_demo_gamma_min,
    )
    env = build_environment(params, seed=args.hardware_demo_seed, quiet=not args.verbose)
    valid_no_c3 = enumerate_assignments(env, require_c3=False)
    feasible = enumerate_assignments(env, require_c3=True)
    if len(feasible) < 2:
        raise RuntimeError("Hardware demo candidate needs at least two feasible states.")

    exact_index, exact = max(enumerate(feasible), key=lambda item: item[1].sum_rate)
    greedy = greedy_assignment(env)
    greedy_polished = local_search(env, greedy) if greedy.feasible else greedy
    qaoa_result, probabilities = run_valid_subspace_qaoa(
        feasible,
        exact_index,
        reps=1,
        grid_steps=args.hardware_demo_grid_steps,
        shots=args.hardware_demo_shots,
        seed=args.hardware_demo_seed,
    )
    qaoa_top = feasible[qaoa_result.top_index]
    local_comparison = compare_qaoa_vs_random_local_search(
        env,
        feasible,
        probabilities,
        exact,
        top_k=args.hardware_demo_top_k,
        random_trials=args.hardware_demo_random_trials,
        seed=args.hardware_demo_seed,
    )
    circuit = build_full_binary_qaoa_circuit(
        env,
        gamma=qaoa_result.gamma,
        beta=qaoa_result.beta,
        reps=args.hardware_demo_reps,
        measure=True,
        coupling_threshold=args.hardware_demo_coupling_threshold,
    )

    return {
        "purpose": (
            "Smaller full-binary QUBO hardware candidate intended to reduce "
            "qubit count and circuit depth before another IBM submission."
        ),
        "scenario": {
            "U": params.U,
            "G": params.G,
            "S": params.S,
            "Nt": params.Nt,
            "n_qubits_full_binary": params.U * params.G,
            "Gamma_min": params.Gamma_min,
            "seed": args.hardware_demo_seed,
            "valid_no_colocation_count": len(valid_no_c3),
            "feasible_assignment_count": len(feasible),
        },
        "exact": exact.to_json(exact.sum_rate),
        "greedy": greedy.to_json(exact.sum_rate),
        "greedy_local_search": greedy_polished.to_json(exact.sum_rate),
        "valid_subspace_qaoa": {
            **asdict(qaoa_result),
            "top_assignment": qaoa_top.to_json(exact.sum_rate),
            "expected_AR_rate": qaoa_result.expected_rate / exact.sum_rate,
            "shots_for_95pct_success": shots_for_success(
                qaoa_result.optimum_probability
            ),
        },
        "qaoa_top_k_local_search": local_comparison.to_json(exact.sum_rate),
        "full_binary_circuit": circuit_summary(circuit),
        "recommended_next_step": (
            "Submit this smaller candidate only after checking transpiled depth "
            "on the selected backend; then compare raw feasibility and projected "
            "AR against the 24-qubit ibm_quebec job."
        ),
    }


def build_neighbor_indices(states: Sequence[AssignmentEval]) -> list[np.ndarray]:
    adjacency = build_assignment_adjacency(states)
    return [np.flatnonzero(row > 0.0) for row in adjacency]


def simulated_annealing_once(
    states: Sequence[AssignmentEval],
    neighbor_indices: Sequence[np.ndarray],
    rng: np.random.Generator,
    *,
    steps: int,
    start_temperature: float,
    end_temperature: float,
) -> AssignmentEval:
    if steps <= 0:
        raise ValueError("Simulated annealing steps must be positive.")
    if start_temperature <= 0.0 or end_temperature <= 0.0:
        raise ValueError("Simulated annealing temperatures must be positive.")

    rates = np.array([row.sum_rate for row in states], dtype=float)
    rate_span = float(rates.max() - rates.min())
    if rate_span <= 0.0:
        return states[int(rng.integers(len(states)))]

    scores = (rates - rates.min()) / rate_span
    current_index = int(rng.integers(len(states)))
    best_index = current_index

    for step in range(steps):
        neighbors = neighbor_indices[current_index]
        if len(neighbors) == 0:
            candidate_index = int(rng.integers(len(states)))
        else:
            candidate_index = int(rng.choice(neighbors))

        if steps == 1:
            temperature = end_temperature
        else:
            progress = step / (steps - 1)
            temperature = start_temperature * (
                end_temperature / start_temperature
            ) ** progress

        delta = float(scores[candidate_index] - scores[current_index])
        if delta >= 0.0 or rng.random() < math.exp(delta / temperature):
            current_index = candidate_index
            if rates[current_index] > rates[best_index]:
                best_index = current_index

    return states[best_index]


def compare_simulated_annealing(
    states: Sequence[AssignmentEval],
    exact: AssignmentEval,
    *,
    restarts: int,
    steps_per_restart: int,
    trials: int,
    seed: int,
    start_temperature: float = 0.25,
    end_temperature: float = 0.01,
) -> SimulatedAnnealingComparison:
    if restarts <= 0:
        raise ValueError("Simulated annealing restarts must be positive.")
    if trials <= 0:
        raise ValueError("Simulated annealing trials must be positive.")

    neighbor_indices = build_neighbor_indices(states)
    rng = np.random.default_rng(seed + 4040)
    trial_ars: list[float] = []
    hits = 0
    overall_best: AssignmentEval | None = None

    for _ in range(trials):
        trial_best: AssignmentEval | None = None
        for _ in range(restarts):
            candidate = simulated_annealing_once(
                states,
                neighbor_indices,
                rng,
                steps=steps_per_restart,
                start_temperature=start_temperature,
                end_temperature=end_temperature,
            )
            if trial_best is None or candidate.sum_rate > trial_best.sum_rate:
                trial_best = candidate

        if trial_best is None:
            raise RuntimeError("Simulated annealing did not produce a candidate.")
        if overall_best is None or trial_best.sum_rate > overall_best.sum_rate:
            overall_best = trial_best
        ar_rate = trial_best.sum_rate / exact.sum_rate
        trial_ars.append(ar_rate)
        if ar_rate >= 1.0 - 1e-9:
            hits += 1

    if overall_best is None:
        raise RuntimeError("Simulated annealing did not produce a best assignment.")

    ar_array = np.array(trial_ars, dtype=float)
    return SimulatedAnnealingComparison(
        restarts=restarts,
        steps_per_restart=steps_per_restart,
        trials=trials,
        start_temperature=start_temperature,
        end_temperature=end_temperature,
        best=overall_best,
        mean_ar=float(ar_array.mean()),
        std_ar=float(ar_array.std(ddof=0)),
        worst_ar=float(ar_array.min()),
        best_ar=float(ar_array.max()),
        optimum_hit_rate=hits / trials,
    )


def shots_for_success(probability: float, target: float = 0.95) -> int:
    if probability <= 0.0:
        return math.inf
    if probability >= target:
        return 1
    return int(math.ceil(math.log(1.0 - target) / math.log(1.0 - probability)))


def run_benchmark(args: argparse.Namespace) -> dict:
    params = SystemParams(
        U=args.uavs,
        G=args.grid_points,
        S=args.survivors,
        Nt=args.antennas,
        Gamma_min=args.gamma_min,
    )
    env = build_environment(params, seed=args.seed, quiet=not args.verbose)
    valid_no_c3 = enumerate_assignments(env, require_c3=False)
    feasible = enumerate_assignments(env, require_c3=True)
    if not feasible:
        raise RuntimeError("No feasible assignments found for this scenario.")

    exact_index, exact = max(enumerate(feasible), key=lambda item: item[1].sum_rate)
    greedy = greedy_assignment(env)
    greedy_polished = local_search(env, greedy)
    qaoa_result, probabilities = run_valid_subspace_qaoa(
        feasible,
        exact_index,
        reps=args.reps,
        grid_steps=args.grid_steps,
        shots=args.shots,
        seed=args.seed,
        record_diagnostics=True,
    )
    top = feasible[qaoa_result.top_index]
    sampled_best = feasible[qaoa_result.sampled_best_index]
    random_baseline = summarize_random_baseline(feasible)
    local_top_k = getattr(args, "local_top_k", 8)
    random_trials = getattr(args, "random_trials", 8)
    sweep_random_trials = getattr(args, "sweep_random_trials", None) or random_trials
    sweep_top_k = parse_int_list(getattr(args, "sweep_top_k", "1,2,4,8,16"))
    local_comparison = compare_qaoa_vs_random_local_search(
        env,
        feasible,
        probabilities,
        exact,
        top_k=local_top_k,
        random_trials=random_trials,
        seed=args.seed,
    )
    simulated_annealing = compare_simulated_annealing(
        feasible,
        exact,
        restarts=getattr(args, "sa_restarts", local_top_k),
        steps_per_restart=getattr(args, "sa_steps", 24),
        trials=getattr(args, "sa_trials", 32),
        seed=args.seed,
        start_temperature=getattr(args, "sa_start_temperature", 0.25),
        end_temperature=getattr(args, "sa_end_temperature", 0.01),
    )
    top_k_sweep = build_top_k_sweep(
        env,
        feasible,
        probabilities,
        exact,
        top_k_values=sweep_top_k,
        random_trials=sweep_random_trials,
        seed=args.seed,
    )
    probability_noise_robustness = build_probability_noise_robustness(
        env,
        feasible,
        probabilities,
        exact,
        top_k=local_top_k,
        shots=args.shots,
        noise_levels=getattr(args, "noise_levels", "0,0.1,0.25,0.5,0.75,1.0"),
    )
    qaoa_success_probability = 1.0 - (1.0 - qaoa_result.optimum_probability) ** args.shots
    random_success_probability = 1.0 - (
        1.0 - random_baseline["uniform_optimum_probability"]
    ) ** args.shots

    ordered_probability = sorted(
        (
            {
                "assignment": list(row.assignment),
                "sum_rate": row.sum_rate,
                "probability": float(probabilities[i]),
                "AR_rate": row.sum_rate / exact.sum_rate,
            }
            for i, row in enumerate(feasible)
        ),
        key=lambda row: row["probability"],
        reverse=True,
    )

    return {
        "scenario": {
            "U": params.U,
            "G": params.G,
            "S": params.S,
            "Nt": params.Nt,
            "n_qubits_full_binary": params.U * params.G,
            "Gamma_min": params.Gamma_min,
            "seed": args.seed,
            "valid_no_colocation_count": len(valid_no_c3),
            "feasible_assignment_count": len(feasible),
            "grid_points_xyz": env.p_grid.round(6).tolist(),
            "survivors_xyz": env.q_surv.round(6).tolist(),
        },
        "exact": exact.to_json(exact.sum_rate),
        "greedy": greedy.to_json(exact.sum_rate),
        "greedy_local_search": greedy_polished.to_json(exact.sum_rate),
        "valid_subspace_qaoa": {
            **asdict(qaoa_result),
            "top_assignment": top.to_json(exact.sum_rate),
            "sampled_best_assignment": sampled_best.to_json(exact.sum_rate),
            "expected_AR_rate": qaoa_result.expected_rate / exact.sum_rate,
            "shot_success_probability": qaoa_success_probability,
            "shots_for_95pct_success": shots_for_success(
                qaoa_result.optimum_probability
            ),
            "expected_samples_to_optimum": 1.0 / qaoa_result.optimum_probability,
            "enrichment_vs_uniform": (
                qaoa_result.optimum_probability
                / random_baseline["uniform_optimum_probability"]
            ),
            "top_probabilities": ordered_probability[: min(args.top_k, len(ordered_probability))],
        },
        "qaoa_top_k_local_search": local_comparison.to_json(exact.sum_rate),
        "simulated_annealing": simulated_annealing.to_json(exact.sum_rate),
        "top_k_sweep": top_k_sweep,
        "probability_noise_robustness": probability_noise_robustness,
        "random_feasible": {
            **random_baseline,
            "mean_AR_rate": random_baseline["mean_rate"] / exact.sum_rate,
            "shot_success_probability": random_success_probability,
            "shots_for_95pct_success": shots_for_success(
                random_baseline["uniform_optimum_probability"]
            ),
            "expected_samples_to_optimum": (
                1.0 / random_baseline["uniform_optimum_probability"]
            ),
        },
        "hardware_evidence": build_hardware_evidence_status(),
    }


def summarize_metric(rows: Sequence[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([row[key] for row in rows]))


def summarize_suite_rows(rows: Sequence[dict]) -> dict:
    return {
        "count": len(rows),
        "mean_feasible_assignments": summarize_metric(rows, "feasible_assignment_count"),
        "mean_greedy_AR_rate": summarize_metric(rows, "greedy_AR_rate"),
        "mean_greedy_local_AR_rate": summarize_metric(rows, "greedy_local_AR_rate"),
        "mean_qaoa_top_AR_rate": summarize_metric(rows, "qaoa_top_AR_rate"),
        "mean_qaoa_raw_top_k_AR_rate": summarize_metric(
            rows, "qaoa_raw_top_k_AR_rate"
        ),
        "mean_qaoa_top_k_local_AR_rate": summarize_metric(
            rows, "qaoa_top_k_local_AR_rate"
        ),
        "mean_qaoa_local_gain_AR_rate": summarize_metric(
            rows, "qaoa_local_gain_AR_rate"
        ),
        "mean_random_raw_top_k_AR_rate": summarize_metric(
            rows, "random_raw_top_k_AR_rate"
        ),
        "mean_random_local_gain_AR_rate": summarize_metric(
            rows, "random_local_gain_AR_rate"
        ),
        "mean_random_top_k_local_AR_rate": summarize_metric(
            rows, "random_top_k_local_AR_rate"
        ),
        "mean_simulated_annealing_AR_rate": summarize_metric(
            rows, "simulated_annealing_AR_rate"
        ),
        "mean_qaoa_optimum_probability": summarize_metric(
            rows, "qaoa_optimum_probability"
        ),
        "simulated_annealing_optimum_hits": int(
            sum(row["simulated_annealing_AR_rate"] >= 1.0 - 1e-9 for row in rows)
        ),
        "qaoa_top_k_optimum_hits": int(
            sum(row["qaoa_top_k_local_AR_rate"] >= 1.0 - 1e-9 for row in rows)
        ),
        "qaoa_top_k_beats_greedy": int(
            sum(row["qaoa_top_k_local_AR_rate"] > row["greedy_AR_rate"] + 1e-9 for row in rows)
        ),
        "qaoa_top_k_beats_random_top_k": int(
            sum(
                row["qaoa_top_k_local_AR_rate"]
                > row["random_top_k_local_AR_rate"] + 1e-9
                for row in rows
            )
        ),
    }


def summarize_suite_top_k_sweep(rows: Sequence[dict]) -> list[dict]:
    top_k_values = sorted(
        {
            sweep_row["top_k"]
            for row in rows
            for sweep_row in row.get("top_k_sweep", [])
        }
    )
    summaries: list[dict] = []
    for top_k in top_k_values:
        sweep_rows = [
            sweep_row
            for row in rows
            for sweep_row in row.get("top_k_sweep", [])
            if sweep_row["top_k"] == top_k
        ]
        if not sweep_rows:
            continue
        summaries.append(
            {
                "top_k": top_k,
                "count": len(sweep_rows),
                "mean_effective_top_k": float(
                    np.mean([row["effective_top_k"] for row in sweep_rows])
                ),
                "mean_qaoa_AR_rate": float(
                    np.mean([row["qaoa_AR_rate"] for row in sweep_rows])
                ),
                "mean_qaoa_raw_AR_rate": float(
                    np.mean([row["qaoa_raw_AR_rate"] for row in sweep_rows])
                ),
                "mean_qaoa_local_gain_AR_rate": float(
                    np.mean([row["qaoa_local_gain_AR_rate"] for row in sweep_rows])
                ),
                "qaoa_optimum_hits": int(
                    sum(row["qaoa_optimum_hit"] for row in sweep_rows)
                ),
                "mean_qaoa_probability_mass": float(
                    np.mean([row["qaoa_probability_mass"] for row in sweep_rows])
                ),
                "mean_random_AR_rate": float(
                    np.mean([row["random_mean_AR_rate"] for row in sweep_rows])
                ),
                "mean_random_raw_AR_rate": float(
                    np.mean([row["random_raw_mean_AR_rate"] for row in sweep_rows])
                ),
                "mean_random_local_gain_AR_rate": float(
                    np.mean(
                        [row["random_local_gain_mean_AR_rate"] for row in sweep_rows]
                    )
                ),
                "mean_random_optimum_hit_rate": float(
                    np.mean([row["random_optimum_hit_rate"] for row in sweep_rows])
                ),
                "mean_random_exact_candidate_hit_probability": float(
                    np.mean(
                        [
                            row["random_exact_candidate_hit_probability"]
                            for row in sweep_rows
                        ]
                    )
                ),
                "mean_qaoa_gain_over_random": float(
                    np.mean([row["qaoa_gain_over_random_mean"] for row in sweep_rows])
                ),
                "qaoa_beats_random_mean_count": int(
                    sum(
                        row["qaoa_AR_rate"] > row["random_mean_AR_rate"] + 1e-9
                        for row in sweep_rows
                    )
                ),
                "mean_qaoa_beats_random_trial_rate": float(
                    np.mean(
                        [row["qaoa_beats_random_trial_rate"] for row in sweep_rows]
                    )
                ),
            }
        )
    return summaries


def parse_seed_range(seed_range: str) -> list[int]:
    if ":" in seed_range:
        start_text, end_text = seed_range.split(":", 1)
        start = int(start_text)
        end = int(end_text)
        return list(range(start, end + 1))
    return [int(part.strip()) for part in seed_range.split(",") if part.strip()]


def run_suite(args: argparse.Namespace) -> dict:
    seeds = parse_seed_range(getattr(args, "suite_seed_range", "1:60"))
    top_k = getattr(args, "suite_top_k", 8)
    random_trials = getattr(args, "suite_random_trials", 8)
    sweep_random_trials = getattr(args, "suite_sweep_random_trials", None) or random_trials
    sweep_top_k = parse_int_list(getattr(args, "suite_sweep_top_k", "1,2,4,8,16"))
    sa_restarts = getattr(args, "suite_sa_restarts", top_k)
    sa_steps = getattr(args, "suite_sa_steps", 24)
    sa_trials = getattr(args, "suite_sa_trials", 16)
    sa_start_temperature = getattr(args, "suite_sa_start_temperature", 0.25)
    sa_end_temperature = getattr(args, "suite_sa_end_temperature", 0.01)
    stress_gap = getattr(args, "suite_stress_gap", 0.10)

    rows: list[dict] = []
    skipped: list[dict] = []
    for seed in seeds:
        params = SystemParams(
            U=getattr(args, "suite_uavs", 3),
            G=getattr(args, "suite_grid_points", 6),
            S=getattr(args, "suite_survivors", 5),
            Nt=getattr(args, "suite_antennas", 4),
            Gamma_min=getattr(args, "suite_gamma_min", 0.5),
        )
        env = build_environment(params, seed=seed, quiet=True)
        feasible = enumerate_assignments(env, require_c3=True)
        if len(feasible) < 2:
            skipped.append({"seed": seed, "reason": "fewer than two feasible states"})
            continue

        exact_index, exact = max(enumerate(feasible), key=lambda item: item[1].sum_rate)
        greedy = greedy_assignment(env)
        if not greedy.feasible:
            skipped.append({"seed": seed, "reason": "greedy infeasible"})
            continue
        greedy_polished = local_search(env, greedy)
        qaoa_result, probabilities = run_valid_subspace_qaoa(
            feasible,
            exact_index,
            reps=1,
            grid_steps=getattr(args, "suite_grid_steps", 31),
            shots=getattr(args, "suite_shots", 1024),
            seed=seed,
        )
        qaoa_top = feasible[qaoa_result.top_index]
        local_comparison = compare_qaoa_vs_random_local_search(
            env,
            feasible,
            probabilities,
            exact,
            top_k=top_k,
            random_trials=random_trials,
            seed=seed,
        )
        simulated_annealing = compare_simulated_annealing(
            feasible,
            exact,
            restarts=sa_restarts,
            steps_per_restart=sa_steps,
            trials=sa_trials,
            seed=seed,
            start_temperature=sa_start_temperature,
            end_temperature=sa_end_temperature,
        )
        top_k_sweep = build_top_k_sweep(
            env,
            feasible,
            probabilities,
            exact,
            top_k_values=sweep_top_k,
            random_trials=sweep_random_trials,
            seed=seed,
        )
        rows.append(
            {
                "seed": seed,
                "feasible_assignment_count": len(feasible),
                "exact_sum_rate": exact.sum_rate,
                "greedy_AR_rate": greedy.sum_rate / exact.sum_rate,
                "greedy_local_AR_rate": greedy_polished.sum_rate / exact.sum_rate,
                "qaoa_top_AR_rate": qaoa_top.sum_rate / exact.sum_rate,
                "qaoa_raw_top_k_AR_rate": (
                    local_comparison.qaoa_raw_best.sum_rate / exact.sum_rate
                ),
                "qaoa_top_k_local_AR_rate": (
                    local_comparison.qaoa_best.sum_rate / exact.sum_rate
                ),
                "qaoa_local_gain_AR_rate": local_comparison.qaoa_local_gain_ar,
                "random_raw_top_k_AR_rate": local_comparison.random_raw_mean_ar,
                "random_local_gain_AR_rate": (
                    local_comparison.random_local_gain_mean_ar
                ),
                "random_top_k_local_AR_rate": local_comparison.random_mean_ar,
                "random_top_k_optimum_hit_rate": (
                    local_comparison.random_optimum_hit_rate
                ),
                "simulated_annealing_AR_rate": simulated_annealing.mean_ar,
                "simulated_annealing_optimum_hit_rate": (
                    simulated_annealing.optimum_hit_rate
                ),
                "simulated_annealing_assignment": list(
                    simulated_annealing.best.assignment
                ),
                "qaoa_optimum_probability": qaoa_result.optimum_probability,
                "qaoa_top_assignment": list(qaoa_top.assignment),
                "qaoa_top_k_assignment": list(local_comparison.qaoa_best.assignment),
                "top_k_sweep": top_k_sweep,
            }
        )

    stress_rows = [row for row in rows if 1.0 - row["greedy_AR_rate"] >= stress_gap]
    return {
        "scenario": {
            "U": getattr(args, "suite_uavs", 3),
            "G": getattr(args, "suite_grid_points", 6),
            "S": getattr(args, "suite_survivors", 5),
            "Nt": getattr(args, "suite_antennas", 4),
            "n_qubits_full_binary": (
                getattr(args, "suite_uavs", 3) * getattr(args, "suite_grid_points", 6)
            ),
            "Gamma_min": getattr(args, "suite_gamma_min", 0.5),
            "seed_range": getattr(args, "suite_seed_range", "1:60"),
            "top_k": top_k,
            "random_trials": random_trials,
            "sweep_top_k": sweep_top_k,
            "sweep_random_trials": sweep_random_trials,
            "simulated_annealing_restarts": sa_restarts,
            "simulated_annealing_steps": sa_steps,
            "simulated_annealing_trials": sa_trials,
            "stress_gap": stress_gap,
        },
        "all": summarize_suite_rows(rows),
        "stress": summarize_suite_rows(stress_rows),
        "top_k_sweep": summarize_suite_top_k_sweep(rows),
        "stress_top_k_sweep": summarize_suite_top_k_sweep(stress_rows),
        "rows": rows,
        "skipped": skipped,
    }


def run_scale_challenge(args: argparse.Namespace) -> dict:
    scale_args = argparse.Namespace(
        uavs=args.scale_uavs,
        grid_points=args.scale_grid_points,
        survivors=args.scale_survivors,
        antennas=args.scale_antennas,
        gamma_min=args.scale_gamma_min,
        seed=args.scale_seed,
        reps=1,
        grid_steps=args.scale_grid_steps,
        shots=args.scale_shots,
        top_k=args.scale_top_k,
        local_top_k=args.scale_top_k,
        random_trials=args.scale_random_trials,
        sweep_top_k=args.scale_sweep_top_k,
        sweep_random_trials=args.scale_sweep_random_trials,
        sa_restarts=args.scale_sa_restarts,
        sa_steps=args.scale_sa_steps,
        sa_trials=args.scale_sa_trials,
        sa_start_temperature=args.scale_sa_start_temperature,
        sa_end_temperature=args.scale_sa_end_temperature,
        noise_levels=args.noise_levels,
        verbose=args.verbose,
    )
    return run_benchmark(scale_args)


def generate_visualizations(results: dict, output_dir: str | Path = "figures") -> list[str]:
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    qaoa = results["valid_subspace_qaoa"]
    random = results["random_feasible"]
    simulated_annealing = results.get("simulated_annealing")
    top_k_sweep = results.get("top_k_sweep", [])
    suite = results.get("suite", {})

    trace = qaoa.get("best_so_far_trace", [])
    if trace:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        evaluations = [row["evaluation"] for row in trace]
        best_ar = [row["best_expected_AR_rate"] for row in trace]
        top_ar = [row["top_AR_rate"] for row in trace]
        best_top_ar = np.maximum.accumulate(np.array(top_ar, dtype=float))
        ax.step(evaluations, best_ar, where="post", label="Best expected AR", color="#1f77b4")
        ax.step(
            evaluations,
            best_top_ar,
            where="post",
            label="Best top-state AR found",
            color="#d62728",
        )
        ax.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.2, label="Exact optimum AR")
        ax.set_xlabel("QAOA grid-search evaluations")
        ax.set_ylabel("AR rate")
        ax.set_title("QAOA parameter-search convergence")
        ax.set_ylim(max(0.0, min(best_ar + top_ar) - 0.03), 1.03)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = output_path / "qaoa_convergence.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(str(path))

    landscape = qaoa.get("angle_landscape", [])
    gamma_grid = qaoa.get("gamma_grid", [])
    beta_grid = qaoa.get("beta_grid", [])
    if landscape and gamma_grid and beta_grid:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        image = ax.imshow(
            np.array(landscape, dtype=float).T,
            origin="lower",
            aspect="auto",
            extent=[
                min(gamma_grid),
                max(gamma_grid),
                min(beta_grid),
                max(beta_grid),
            ],
            cmap="viridis",
        )
        ax.scatter([qaoa["gamma"]], [qaoa["beta"]], marker="*", s=140, color="white", edgecolor="black", label="Selected angles")
        ax.set_xlabel("gamma")
        ax.set_ylabel("beta")
        ax.set_title("QAOA expected-AR landscape")
        fig.colorbar(image, ax=ax, label="Expected AR rate")
        ax.legend(loc="upper right")
        fig.tight_layout()
        path = output_path / "qaoa_angle_landscape.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(str(path))

    probability_rows = qaoa.get("top_probabilities", [])
    if probability_rows:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        labels = [str(row["assignment"]) for row in probability_rows]
        probabilities = [row["probability"] for row in probability_rows]
        colors = [
            "#2ca02c" if row["AR_rate"] >= 1.0 - 1e-9 else "#1f77b4"
            for row in probability_rows
        ]
        ax.bar(labels, probabilities, color=colors, alpha=0.9)
        ax.axhline(
            random["uniform_optimum_probability"],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Uniform feasible optimum probability",
        )
        ax.set_xlabel("Assignment")
        ax.set_ylabel("Probability")
        ax.set_title("QAOA probability concentration on high-rate assignments")
        ax.tick_params(axis="x", rotation=55)
        ax.legend()
        fig.tight_layout()
        path = output_path / "qaoa_probability_distribution.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(str(path))

    if top_k_sweep or suite.get("stress_top_k_sweep"):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
        if top_k_sweep:
            k_values = [row["top_k"] for row in top_k_sweep]
            qaoa_ar = [row["qaoa_AR_rate"] for row in top_k_sweep]
            random_ar = [row["random_mean_AR_rate"] for row in top_k_sweep]
            random_hit = [row["random_optimum_hit_rate"] for row in top_k_sweep]
            axes[0].plot(k_values, qaoa_ar, marker="o", color="#17becf", label="QAOA top-K + local")
            axes[0].plot(k_values, random_ar, marker="s", color="#9467bd", label="Random top-K + local")
            axes[0].plot(k_values, random_hit, marker="^", color="#ff7f0e", label="Random optimum-hit rate")
            if simulated_annealing:
                axes[0].axhline(
                    simulated_annealing["mean_AR_rate"],
                    color="#8c564b",
                    linestyle="--",
                    linewidth=1.4,
                    label="SA mean AR",
                )
            axes[0].set_title("Headline candidate efficiency")
            axes[0].set_xlabel("Candidate budget K")
            axes[0].set_ylabel("AR rate / hit rate")
            axes[0].set_ylim(0.0, 1.04)
            axes[0].grid(alpha=0.25)
            axes[0].legend()

        stress_sweep = suite.get("stress_top_k_sweep", [])
        if stress_sweep:
            k_values = [row["top_k"] for row in stress_sweep]
            qaoa_ar = [row["mean_qaoa_AR_rate"] for row in stress_sweep]
            random_ar = [row["mean_random_AR_rate"] for row in stress_sweep]
            qaoa_hits = [
                row["qaoa_optimum_hits"] / row["count"] for row in stress_sweep
            ]
            stress_sa = suite.get("stress", {}).get(
                "mean_simulated_annealing_AR_rate"
            )
            axes[1].plot(k_values, qaoa_ar, marker="o", color="#17becf", label="QAOA mean AR")
            axes[1].plot(k_values, random_ar, marker="s", color="#9467bd", label="Random mean AR")
            axes[1].plot(k_values, qaoa_hits, marker="^", color="#2ca02c", label="QAOA optimum-hit rate")
            if stress_sa is not None:
                axes[1].axhline(
                    stress_sa,
                    color="#8c564b",
                    linestyle="--",
                    linewidth=1.4,
                    label="SA mean AR",
                )
            axes[1].set_title("Stress-suite candidate efficiency")
            axes[1].set_xlabel("Candidate budget K")
            axes[1].set_ylabel("AR rate / hit rate")
            axes[1].set_ylim(0.0, 1.04)
            axes[1].grid(alpha=0.25)
            axes[1].legend()

        fig.tight_layout()
        path = output_path / "qaoa_candidate_efficiency.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(str(path))

    scenario = results["scenario"]
    grid_points = np.array(scenario.get("grid_points_xyz", []), dtype=float)
    survivors = np.array(scenario.get("survivors_xyz", []), dtype=float)
    if grid_points.size and survivors.size:
        exact_assignment = results["exact"]["assignment"]
        greedy_assignment = results["greedy"]["assignment"]
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        ax.scatter(grid_points[:, 0], grid_points[:, 1], marker="s", s=60, color="#7f7f7f", label="Candidate UAV grid")
        ax.scatter(survivors[:, 0], survivors[:, 1], marker="*", s=120, color="#d62728", label="Survivors")
        exact_points = grid_points[exact_assignment]
        greedy_points = grid_points[greedy_assignment]
        ax.scatter(greedy_points[:, 0], greedy_points[:, 1], marker="x", s=95, color="#ff7f0e", label="Greedy UAVs")
        ax.scatter(exact_points[:, 0], exact_points[:, 1], marker="o", s=95, facecolor="none", edgecolor="#17becf", linewidth=2.2, label="QAOA/exact UAVs")
        for index, point in enumerate(grid_points):
            ax.text(point[0] + 4, point[1] + 4, f"g{index}", fontsize=8)
        for index, point in enumerate(survivors):
            ax.text(point[0] + 4, point[1] + 4, f"s{index}", fontsize=8, color="#8c0000")
        for uav_index, grid_index in enumerate(exact_assignment):
            point = grid_points[grid_index]
            ax.text(
                point[0] + 8,
                point[1] - 10,
                f"Q:u{uav_index}",
                fontsize=8,
                color="#008c99",
            )
        for uav_index, grid_index in enumerate(greedy_assignment):
            point = grid_points[grid_index]
            ax.text(
                point[0] + 8,
                point[1] - 22,
                f"G:u{uav_index}",
                fontsize=8,
                color="#b35a00",
            )
        if set(exact_assignment) == set(greedy_assignment):
            ax.text(
                0.02,
                0.02,
                "Same occupied grid set; UAV identity differs.",
                transform=ax.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
            )
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        ax.set_title("UAV deployment map: greedy vs QAOA/exact")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        path = output_path / "qaoa_deployment_map.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(str(path))

    return saved


def print_summary(results: dict) -> None:
    scenario = results["scenario"]
    exact = results["exact"]
    greedy = results["greedy"]
    polished = results["greedy_local_search"]
    qaoa = results["valid_subspace_qaoa"]
    qaoa_local = results["qaoa_top_k_local_search"]
    simulated_annealing = results["simulated_annealing"]
    random = results["random_feasible"]

    print("Hard QAOA-ISAC benchmark")
    print(
        "Scenario: U={U}, G={G}, S={S}, n={n_qubits_full_binary}, "
        "Gamma_min={Gamma_min}, seed={seed}".format(**scenario)
    )
    print(
        "Valid assignments: {feasible_assignment_count} feasible of "
        "{valid_no_colocation_count} no-colocation states".format(**scenario)
    )
    print()
    print("Method                  Rate (Mbps)   AR_rate   Assignment")
    print("Exact enumeration       {0:10.3f}   {1:7.3f}   {2}".format(
        exact["sum_rate"] / 1e6,
        exact["AR_rate"],
        exact["assignment"],
    ))
    print("Greedy                  {0:10.3f}   {1:7.3f}   {2}".format(
        greedy["sum_rate"] / 1e6,
        greedy["AR_rate"],
        greedy["assignment"],
    ))
    print("Greedy + local search   {0:10.3f}   {1:7.3f}   {2}".format(
        polished["sum_rate"] / 1e6,
        polished["AR_rate"],
        polished["assignment"],
    ))
    print("Random feasible mean    {0:10.3f}   {1:7.3f}   -".format(
        random["mean_rate"] / 1e6,
        random["mean_AR_rate"],
    ))
    print("Valid-subspace QAOA top {0:10.3f}   {1:7.3f}   {2}".format(
        qaoa["top_assignment"]["sum_rate"] / 1e6,
        qaoa["top_assignment"]["AR_rate"],
        qaoa["top_assignment"]["assignment"],
    ))
    print("QAOA top-{0} + local    {1:10.3f}   {2:7.3f}   {3}".format(
        qaoa_local["top_k"],
        qaoa_local["qaoa_best"]["sum_rate"] / 1e6,
        qaoa_local["qaoa_best"]["AR_rate"],
        qaoa_local["qaoa_best"]["assignment"],
    ))
    print("Simulated annealing     {0:10.3f}   {1:7.3f}   {2}".format(
        simulated_annealing["best"]["sum_rate"] / 1e6,
        simulated_annealing["best"]["AR_rate"],
        simulated_annealing["best"]["assignment"],
    ))
    print()
    print(
        "QAOA optimum probability: {0:.3f} "
        "({1:.1f}x uniform), 95% success in {2} shots vs {3} uniform shots".format(
            qaoa["optimum_probability"],
            qaoa["enrichment_vs_uniform"],
            qaoa["shots_for_95pct_success"],
            random["shots_for_95pct_success"],
        )
    )
    print(
        "QAOA top-{0}+local AR={1:.3f}; random top-{0}+local mean AR={2:.3f}".format(
            qaoa_local["top_k"],
            qaoa_local["qaoa_best"]["AR_rate"],
            qaoa_local["random_mean_AR_rate"],
        )
    )
    if "qaoa_raw_best" in qaoa_local:
        print(
            "Local attribution: QAOA raw top-{0} AR={1:.3f}, local gain={2:.3f}; "
            "random raw mean AR={3:.3f}, random local gain={4:.3f}".format(
                qaoa_local["top_k"],
                qaoa_local["qaoa_raw_best"]["AR_rate"],
                qaoa_local["qaoa_local_gain_AR_rate"],
                qaoa_local["random_raw_mean_AR_rate"],
                qaoa_local["random_local_gain_mean_AR_rate"],
            )
        )
    print(
        "Simulated annealing mean AR={0:.3f}; optimum-hit rate={1:.3f}".format(
            simulated_annealing["mean_AR_rate"],
            simulated_annealing["optimum_hit_rate"],
        )
    )
    print(
        "Greedy gap: {0:.1%}; QAOA top gap: {1:.1%}".format(
            1.0 - greedy["AR_rate"],
            1.0 - qaoa["top_assignment"]["AR_rate"],
        )
    )

    top_k_sweep = results.get("top_k_sweep", [])
    if top_k_sweep:
        print()
        print("Top-K candidate efficiency sweep")
        print("K    QAOA AR   Random AR   QAOA gain   QAOA hit   Random hit")
        for row in top_k_sweep:
            print(
                "{top_k:<4d} {qaoa_AR_rate:8.3f}   {random_mean_AR_rate:9.3f}   "
                "{qaoa_gain_over_random_mean:9.3f}   {qaoa_hit:8s}   "
                "{random_optimum_hit_rate:10.3f}".format(
                    **row,
                    qaoa_hit="yes" if row["qaoa_optimum_hit"] else "no",
                )
            )

    robustness = results.get("probability_noise_robustness", {}).get("rows", [])
    if robustness:
        print()
        print("Probability-noise robustness")
        print("blend   optimum p   QAOA top-k+local AR   shots@95%")
        for row in robustness:
            print(
                "{uniform_blend:5.2f}   {optimum_probability:9.4f}   "
                "{top_k_local_AR_rate:19.3f}   {shots_for_95pct_success}".format(
                    **row
                )
            )

    suite = results.get("suite")
    if suite:
        scenario = suite["scenario"]
        all_rows = suite["all"]
        stress = suite["stress"]
        print()
        print(
            "Suite: U={U}, G={G}, S={S}, seeds={seed_range}, top_k={top_k}".format(
                **scenario
            )
        )
        print(
            "Evaluated seeds ({count}): greedy AR={mean_greedy_AR_rate:.3f}, "
            "greedy+local AR={mean_greedy_local_AR_rate:.3f}, "
            "QAOA raw top-k AR={mean_qaoa_raw_top_k_AR_rate:.3f}, "
            "QAOA top-k+local AR={mean_qaoa_top_k_local_AR_rate:.3f}, "
            "QAOA local gain={mean_qaoa_local_gain_AR_rate:.3f}, "
            "random top-k+local AR={mean_random_top_k_local_AR_rate:.3f}, "
            "SA AR={mean_simulated_annealing_AR_rate:.3f}".format(
                **all_rows
            )
        )
        print(
            "Stress seeds ({count}): greedy AR={mean_greedy_AR_rate:.3f}, "
            "greedy+local AR={mean_greedy_local_AR_rate:.3f}, "
            "QAOA raw top-k AR={mean_qaoa_raw_top_k_AR_rate:.3f}, "
            "QAOA top-k+local AR={mean_qaoa_top_k_local_AR_rate:.3f}, "
            "QAOA local gain={mean_qaoa_local_gain_AR_rate:.3f}, "
            "random top-k+local AR={mean_random_top_k_local_AR_rate:.3f}, "
            "SA AR={mean_simulated_annealing_AR_rate:.3f}".format(
                **stress
            )
        )
        suite_sweep = suite.get("top_k_sweep", [])
        if suite_sweep:
            print("Suite top-K sweep: K, effective K, QAOA AR, random AR, QAOA hits, gain")
            for row in suite_sweep:
                print(
                    "  {top_k} ({mean_effective_top_k:.1f}): {mean_qaoa_AR_rate:.3f}, "
                    "{mean_random_AR_rate:.3f}, {qaoa_optimum_hits}/{count}, "
                    "{mean_qaoa_gain_over_random:.3f}".format(**row)
                )
        stress_sweep = suite.get("stress_top_k_sweep", [])
        if stress_sweep:
            print("Stress top-K sweep: K, effective K, QAOA AR, random AR, QAOA hits, gain")
            for row in stress_sweep:
                print(
                    "  {top_k} ({mean_effective_top_k:.1f}): {mean_qaoa_AR_rate:.3f}, "
                    "{mean_random_AR_rate:.3f}, {qaoa_optimum_hits}/{count}, "
                    "{mean_qaoa_gain_over_random:.3f}".format(**row)
                )

    scale = results.get("scale_challenge")
    if scale:
        scenario = scale["scenario"]
        scale_qaoa_local = scale["qaoa_top_k_local_search"]
        print()
        print(
            "Scale challenge: U={U}, G={G}, S={S}, feasible={feasible_assignment_count}, "
            "full-binary qubits={n_qubits_full_binary}".format(**scenario)
        )
        print(
            "Scale challenge AR: greedy={0:.3f}, QAOA raw top-{1}={2:.3f}, "
            "QAOA top-{1}+local={3:.3f}, SA mean={4:.3f}".format(
                scale["greedy"]["AR_rate"],
                scale_qaoa_local["top_k"],
                scale_qaoa_local["qaoa_raw_best"]["AR_rate"],
                scale_qaoa_local["qaoa_best"]["AR_rate"],
                scale["simulated_annealing"]["mean_AR_rate"],
            )
        )

    hardware_demo = results.get("hardware_demo_candidate")
    if hardware_demo:
        scenario = hardware_demo["scenario"]
        circuit = hardware_demo["full_binary_circuit"]
        qaoa = hardware_demo["valid_subspace_qaoa"]
        qaoa_local = hardware_demo["qaoa_top_k_local_search"]
        print()
        print(
            "Hardware demo candidate: U={U}, G={G}, S={S}, feasible={feasible_assignment_count}, "
            "full-binary qubits={n_qubits_full_binary}".format(**scenario)
        )
        print(
            "Hardware demo circuit: depth={depth}, two-qubit gates={two_qubit_gate_count}, ops={ops}".format(
                **circuit
            )
        )
        print(
            "Hardware demo AR: greedy={0:.3f}, QAOA top={1:.3f}, "
            "QAOA top-{2}+local={3:.3f}, optimum probability={4:.3f}".format(
                hardware_demo["greedy"]["AR_rate"],
                qaoa["top_assignment"]["AR_rate"],
                qaoa_local["top_k"],
                qaoa_local["qaoa_best"]["AR_rate"],
                qaoa["optimum_probability"],
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uavs", type=int, default=4)
    parser.add_argument("--grid-points", type=int, default=6)
    parser.add_argument("--survivors", type=int, default=6)
    parser.add_argument("--antennas", type=int, default=4)
    parser.add_argument("--gamma-min", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--grid-steps", type=int, default=61)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--local-top-k", type=int, default=8)
    parser.add_argument("--random-trials", type=int, default=8)
    parser.add_argument("--sweep-top-k", default="1,2,4,8,16")
    parser.add_argument("--sweep-random-trials", type=int, default=None)
    parser.add_argument("--noise-levels", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--sa-restarts", type=int, default=8)
    parser.add_argument("--sa-steps", type=int, default=24)
    parser.add_argument("--sa-trials", type=int, default=32)
    parser.add_argument("--sa-start-temperature", type=float, default=0.25)
    parser.add_argument("--sa-end-temperature", type=float, default=0.01)
    parser.add_argument("--include-suite", action="store_true")
    parser.add_argument("--suite-seed-range", default="1:60")
    parser.add_argument("--suite-uavs", type=int, default=3)
    parser.add_argument("--suite-grid-points", type=int, default=6)
    parser.add_argument("--suite-survivors", type=int, default=5)
    parser.add_argument("--suite-antennas", type=int, default=4)
    parser.add_argument("--suite-gamma-min", type=float, default=0.5)
    parser.add_argument("--suite-grid-steps", type=int, default=31)
    parser.add_argument("--suite-shots", type=int, default=1024)
    parser.add_argument("--suite-top-k", type=int, default=8)
    parser.add_argument("--suite-random-trials", type=int, default=8)
    parser.add_argument("--suite-sweep-top-k", default="1,2,4,8,16")
    parser.add_argument("--suite-sweep-random-trials", type=int, default=None)
    parser.add_argument("--suite-sa-restarts", type=int, default=8)
    parser.add_argument("--suite-sa-steps", type=int, default=24)
    parser.add_argument("--suite-sa-trials", type=int, default=16)
    parser.add_argument("--suite-sa-start-temperature", type=float, default=0.25)
    parser.add_argument("--suite-sa-end-temperature", type=float, default=0.01)
    parser.add_argument("--suite-stress-gap", type=float, default=0.10)
    parser.add_argument("--include-scale-challenge", action="store_true")
    parser.add_argument("--scale-uavs", type=int, default=4)
    parser.add_argument("--scale-grid-points", type=int, default=7)
    parser.add_argument("--scale-survivors", type=int, default=7)
    parser.add_argument("--scale-antennas", type=int, default=4)
    parser.add_argument("--scale-gamma-min", type=float, default=0.5)
    parser.add_argument("--scale-seed", type=int, default=8)
    parser.add_argument("--scale-grid-steps", type=int, default=31)
    parser.add_argument("--scale-shots", type=int, default=1024)
    parser.add_argument("--scale-top-k", type=int, default=8)
    parser.add_argument("--scale-random-trials", type=int, default=32)
    parser.add_argument("--scale-sweep-top-k", default="1,2,4,8,16")
    parser.add_argument("--scale-sweep-random-trials", type=int, default=32)
    parser.add_argument("--scale-sa-restarts", type=int, default=8)
    parser.add_argument("--scale-sa-steps", type=int, default=24)
    parser.add_argument("--scale-sa-trials", type=int, default=16)
    parser.add_argument("--scale-sa-start-temperature", type=float, default=0.25)
    parser.add_argument("--scale-sa-end-temperature", type=float, default=0.01)
    parser.add_argument("--include-hardware-demo-candidate", action="store_true")
    parser.add_argument("--hardware-demo-uavs", type=int, default=3)
    parser.add_argument("--hardware-demo-grid-points", type=int, default=6)
    parser.add_argument("--hardware-demo-survivors", type=int, default=5)
    parser.add_argument("--hardware-demo-antennas", type=int, default=4)
    parser.add_argument("--hardware-demo-gamma-min", type=float, default=0.5)
    parser.add_argument("--hardware-demo-seed", type=int, default=35)
    parser.add_argument("--hardware-demo-grid-steps", type=int, default=31)
    parser.add_argument("--hardware-demo-shots", type=int, default=1024)
    parser.add_argument("--hardware-demo-top-k", type=int, default=4)
    parser.add_argument("--hardware-demo-random-trials", type=int, default=32)
    parser.add_argument("--hardware-demo-reps", type=int, default=1)
    parser.add_argument("--hardware-demo-coupling-threshold", type=float, default=0.0)
    parser.add_argument("--output", default="qaoa_isac_benchmark_results.json")
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(args)
    if args.include_suite:
        results["suite"] = run_suite(args)
    if args.include_scale_challenge:
        results["scale_challenge"] = run_scale_challenge(args)
    if args.include_hardware_demo_candidate:
        results["hardware_demo_candidate"] = run_hardware_demo_candidate(args)
    if args.make_figures:
        results["figures"] = generate_visualizations(results, args.figure_dir)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print()
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
