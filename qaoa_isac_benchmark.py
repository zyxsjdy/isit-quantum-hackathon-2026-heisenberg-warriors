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
    qaoa_best: AssignmentEval
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
            "qaoa_best": self.qaoa_best.to_json(exact_rate),
            "random_mean_AR_rate": self.random_mean_ar,
            "random_std_AR_rate": self.random_std_ar,
            "random_worst_AR_rate": self.random_worst_ar,
            "random_best_AR_rate": self.random_best_ar,
            "random_optimum_hit_rate": self.random_optimum_hit_rate,
            "qaoa_beats_random_trial_rate": self.qaoa_beats_random_trial_rate,
            "random_trials": self.random_trials,
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
    qaoa_best = best_local_search_from_candidates(env, qaoa_candidates)
    qaoa_ar = qaoa_best.sum_rate / exact.sum_rate

    rng = np.random.default_rng(seed + 2026)
    random_ars: list[float] = []
    random_hits = 0
    random_losses = 0
    for _ in range(random_trials):
        size = min(top_k, len(states))
        random_indices = rng.choice(len(states), size=size, replace=False)
        random_candidates = [states[int(i)] for i in random_indices]
        random_best = best_local_search_from_candidates(env, random_candidates)
        ar_rate = random_best.sum_rate / exact.sum_rate
        random_ars.append(ar_rate)
        if ar_rate >= 1.0 - 1e-9:
            random_hits += 1
        if qaoa_ar > ar_rate + 1e-9:
            random_losses += 1

    random_array = np.array(random_ars, dtype=float)

    return LocalSearchComparison(
        top_k=top_k,
        qaoa_best=qaoa_best,
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
                "qaoa_AR_rate": qaoa_ar,
                "qaoa_optimum_hit": qaoa_ar >= 1.0 - 1e-9,
                "qaoa_assignment": list(comparison.qaoa_best.assignment),
                "qaoa_probability_mass": float(probabilities[qaoa_top_indices].sum()),
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
    top_k_sweep = build_top_k_sweep(
        env,
        feasible,
        probabilities,
        exact,
        top_k_values=sweep_top_k,
        random_trials=sweep_random_trials,
        seed=args.seed,
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
        "top_k_sweep": top_k_sweep,
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
        "mean_qaoa_top_k_local_AR_rate": summarize_metric(
            rows, "qaoa_top_k_local_AR_rate"
        ),
        "mean_random_top_k_local_AR_rate": summarize_metric(
            rows, "random_top_k_local_AR_rate"
        ),
        "mean_qaoa_optimum_probability": summarize_metric(
            rows, "qaoa_optimum_probability"
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
                "qaoa_optimum_hits": int(
                    sum(row["qaoa_optimum_hit"] for row in sweep_rows)
                ),
                "mean_qaoa_probability_mass": float(
                    np.mean([row["qaoa_probability_mass"] for row in sweep_rows])
                ),
                "mean_random_AR_rate": float(
                    np.mean([row["random_mean_AR_rate"] for row in sweep_rows])
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
                "qaoa_top_k_local_AR_rate": (
                    local_comparison.qaoa_best.sum_rate / exact.sum_rate
                ),
                "random_top_k_local_AR_rate": local_comparison.random_mean_ar,
                "random_top_k_optimum_hit_rate": (
                    local_comparison.random_optimum_hit_rate
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
            "stress_gap": stress_gap,
        },
        "all": summarize_suite_rows(rows),
        "stress": summarize_suite_rows(stress_rows),
        "top_k_sweep": summarize_suite_top_k_sweep(rows),
        "stress_top_k_sweep": summarize_suite_top_k_sweep(stress_rows),
        "rows": rows,
        "skipped": skipped,
    }


def generate_visualizations(results: dict, output_dir: str | Path = "figures") -> list[str]:
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    qaoa = results["valid_subspace_qaoa"]
    random = results["random_feasible"]
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
            axes[1].plot(k_values, qaoa_ar, marker="o", color="#17becf", label="QAOA mean AR")
            axes[1].plot(k_values, random_ar, marker="s", color="#9467bd", label="Random mean AR")
            axes[1].plot(k_values, qaoa_hits, marker="^", color="#2ca02c", label="QAOA optimum-hit rate")
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
            "QAOA top-k+local AR={mean_qaoa_top_k_local_AR_rate:.3f}, "
            "random top-k+local AR={mean_random_top_k_local_AR_rate:.3f}".format(
                **all_rows
            )
        )
        print(
            "Stress seeds ({count}): greedy AR={mean_greedy_AR_rate:.3f}, "
            "greedy+local AR={mean_greedy_local_AR_rate:.3f}, "
            "QAOA top-k+local AR={mean_qaoa_top_k_local_AR_rate:.3f}, "
            "random top-k+local AR={mean_random_top_k_local_AR_rate:.3f}".format(
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
    parser.add_argument("--suite-stress-gap", type=float, default=0.10)
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
    if args.make_figures:
        results["figures"] = generate_visualizations(results, args.figure_dir)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print()
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
