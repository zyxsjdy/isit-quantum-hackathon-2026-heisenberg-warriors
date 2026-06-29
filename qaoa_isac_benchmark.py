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
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Sequence

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
    gamma: float
    beta: float
    expected_rate: float
    top_index: int
    top_probability: float
    optimum_probability: float
    sampled_best_index: int
    sampled_best_count: int
    shots: int


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
) -> tuple[QAOASubspaceResult, np.ndarray]:
    if reps != 1:
        raise ValueError("This benchmark currently implements the p=1 subspace solver.")
    if len(states) < 2:
        raise ValueError("At least two feasible states are required for QAOA.")

    rates = np.array([row.sum_rate for row in states], dtype=float)
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

    for gamma in np.linspace(0.0, 2.0 * math.pi, grid_steps):
        for beta in np.linspace(0.0, math.pi, grid_steps):
            expected_rate, probabilities = evaluate_angles(float(gamma), float(beta))
            top_index = int(np.argmax(probabilities))
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

    rng = np.random.default_rng(seed)
    sample_counts = rng.multinomial(shots, best_probabilities)
    sampled_indices = np.flatnonzero(sample_counts)
    sampled_best_index = int(max(sampled_indices, key=lambda i: states[int(i)].sum_rate))
    top_index = int(np.argmax(best_probabilities))

    result = QAOASubspaceResult(
        reps=reps,
        grid_steps=grid_steps,
        gamma=best_gamma,
        beta=best_beta,
        expected_rate=float(best_expected),
        top_index=top_index,
        top_probability=float(best_probabilities[top_index]),
        optimum_probability=float(best_probabilities[exact_best_index]),
        sampled_best_index=sampled_best_index,
        sampled_best_count=int(sample_counts[sampled_best_index]),
        shots=shots,
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
    )
    top = feasible[qaoa_result.top_index]
    sampled_best = feasible[qaoa_result.sampled_best_index]
    random_baseline = summarize_random_baseline(feasible)
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
    }


def print_summary(results: dict) -> None:
    scenario = results["scenario"]
    exact = results["exact"]
    greedy = results["greedy"]
    polished = results["greedy_local_search"]
    qaoa = results["valid_subspace_qaoa"]
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
        "Greedy gap: {0:.1%}; QAOA top gap: {1:.1%}".format(
            1.0 - greedy["AR_rate"],
            1.0 - qaoa["top_assignment"]["AR_rate"],
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uavs", type=int, default=3)
    parser.add_argument("--grid-points", type=int, default=6)
    parser.add_argument("--survivors", type=int, default=5)
    parser.add_argument("--antennas", type=int, default=4)
    parser.add_argument("--gamma-min", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--grid-steps", type=int, default=61)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default="qaoa_isac_benchmark_results.json")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(args)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print()
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
