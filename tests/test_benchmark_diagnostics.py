import unittest
from argparse import Namespace

import numpy as np

from qaoa_isac_benchmark import (
    build_full_binary_qaoa_circuit,
    build_environment,
    build_smaller_hardware_evidence_status,
    build_probability_noise_robustness,
    compare_qaoa_vs_random_local_search,
    circuit_summary,
    enumerate_assignments,
    extract_sampler_counts,
    parse_float_list,
    run_suite,
    run_scale_challenge,
    run_valid_subspace_qaoa,
    summarize_full_binary_angle_probe,
    summarize_random_bitstring_projection_baseline,
)
from qaoa_isac_env import SystemParams


class BenchmarkDiagnosticsTest(unittest.TestCase):
    def small_case(self):
        params = SystemParams(U=2, G=4, S=2, Nt=2, Gamma_min=0.0)
        env = build_environment(params, seed=3, quiet=True)
        states = enumerate_assignments(env, require_c3=True)
        self.assertGreaterEqual(len(states), 2)
        exact_index, exact = max(enumerate(states), key=lambda item: item[1].sum_rate)
        probabilities = np.full(len(states), 0.5 / (len(states) - 1), dtype=float)
        probabilities[exact_index] = 0.5
        return env, states, exact, probabilities

    def test_parse_float_list_deduplicates_and_validates(self):
        self.assertEqual(parse_float_list("0,0.25,0.25,1"), [0.0, 0.25, 1.0])
        with self.assertRaises(ValueError):
            parse_float_list("0,-0.1")
        with self.assertRaises(ValueError):
            parse_float_list("1.1")

    def test_local_search_comparison_reports_raw_and_polished_quality(self):
        env, states, exact, probabilities = self.small_case()

        comparison = compare_qaoa_vs_random_local_search(
            env,
            states,
            probabilities,
            exact,
            top_k=2,
            random_trials=4,
            seed=11,
        )

        self.assertGreaterEqual(
            comparison.qaoa_best.sum_rate,
            comparison.qaoa_raw_best.sum_rate - 1e-9,
        )
        self.assertGreaterEqual(comparison.qaoa_local_gain_ar, -1e-9)
        payload = comparison.to_json(exact.sum_rate)
        self.assertIn("qaoa_raw_best", payload)
        self.assertIn("random_raw_mean_AR_rate", payload)
        self.assertIn("random_local_gain_mean_AR_rate", payload)

    def test_probability_noise_blend_moves_optimum_probability_to_uniform(self):
        env, states, exact, probabilities = self.small_case()

        robustness = build_probability_noise_robustness(
            env,
            states,
            probabilities,
            exact,
            top_k=2,
            shots=100,
            noise_levels=[0.0, 1.0],
        )

        rows = robustness["rows"]
        self.assertEqual([row["uniform_blend"] for row in rows], [0.0, 1.0])
        optimum_count = sum(row.sum_rate >= exact.sum_rate - 1e-9 for row in states)
        self.assertAlmostEqual(rows[1]["optimum_probability"], optimum_count / len(states))
        self.assertGreaterEqual(rows[0]["top_k_local_AR_rate"], 0.0)

    def test_extract_sampler_counts_uses_available_classical_register(self):
        class FakeBitArray:
            def get_counts(self):
                return {"10": 3, "01": 2}

        class FakeDataBin:
            c = FakeBitArray()

            def keys(self):
                return ["c"]

        class FakePubResult:
            data = FakeDataBin()

        counts, register = extract_sampler_counts([FakePubResult()], return_register=True)

        self.assertEqual(register, "c")
        self.assertEqual(counts, {"10": 3, "01": 2})

    def test_extract_sampler_counts_keeps_meas_register_support(self):
        class FakeBitArray:
            def get_counts(self):
                return {"11": 4}

        class FakeDataBin:
            meas = FakeBitArray()

            def keys(self):
                return ["meas"]

        class FakePubResult:
            data = FakeDataBin()

        self.assertEqual(extract_sampler_counts([FakePubResult()]), {"11": 4})

    def test_full_binary_circuit_summary_reports_hardware_shape(self):
        params = SystemParams(U=2, G=3, S=2, Nt=2, Gamma_min=0.0)
        env = build_environment(params, seed=5, quiet=True)

        circuit = build_full_binary_qaoa_circuit(env, gamma=0.2, beta=0.3)
        summary = circuit_summary(circuit)

        self.assertEqual(summary["num_qubits"], 6)
        self.assertGreater(summary["depth"], 0)
        self.assertIn("measure", summary["ops"])

    def test_random_bitstring_projection_baseline_is_shot_matched(self):
        env, states, exact, _ = self.small_case()

        baseline = summarize_random_bitstring_projection_baseline(
            states,
            env,
            exact.sum_rate,
            shots=32,
            random_trials=5,
            seed=13,
        )

        self.assertEqual(
            baseline["model"],
            "uniform_full_binary_bitstrings_then_nearest_feasible_projection",
        )
        self.assertEqual(baseline["n_qubits"], env.n_qubits)
        self.assertEqual(baseline["shots_per_trial"], 32)
        self.assertEqual(baseline["random_trials"], 5)
        self.assertGreaterEqual(baseline["projected_optimum_rate_mean"], 0.0)
        self.assertLessEqual(baseline["projected_optimum_rate_mean"], 1.0)
        self.assertGreaterEqual(baseline["projected_best_AR_rate_mean"], 0.0)
        self.assertLessEqual(baseline["projected_best_AR_rate_mean"], 1.0)
        self.assertGreaterEqual(
            baseline["projected_best_AR_rate_mean"],
            baseline["projected_mean_AR_rate_mean"],
        )

    def test_full_binary_angle_probe_reports_projected_probabilities(self):
        env, states, exact, _ = self.small_case()

        probe = summarize_full_binary_angle_probe(
            env,
            states,
            exact,
            reference_gamma=0.2,
            reference_beta=0.3,
            grid_steps=3,
        )

        self.assertEqual(probe["model"], "statevector_p1_full_binary_qubo_grid")
        self.assertEqual(probe["grid_steps"], 3)
        self.assertEqual(probe["evaluations"], 9)
        self.assertIn("reference_angles", probe)
        self.assertIn("qubo_energy_optimized_angles", probe)
        for key in ("reference_angles", "qubo_energy_optimized_angles"):
            self.assertGreaterEqual(probe[key]["raw_feasible_probability"], 0.0)
            self.assertLessEqual(probe[key]["raw_feasible_probability"], 1.0)
            self.assertGreaterEqual(probe[key]["projected_optimum_probability"], 0.0)
            self.assertLessEqual(probe[key]["projected_optimum_probability"], 1.0)
            self.assertLessEqual(probe[key]["projected_best_AR_rate"], 1.0)

    def test_qaoa_can_train_for_top_k_candidate_quality(self):
        _, states, _, _ = self.small_case()
        exact_index, _ = max(enumerate(states), key=lambda item: item[1].sum_rate)

        result, probabilities = run_valid_subspace_qaoa(
            states,
            exact_index,
            reps=1,
            grid_steps=5,
            shots=32,
            seed=7,
            selection_objective="top_k_raw_rate",
            candidate_top_k=2,
        )

        self.assertEqual(result.selection_objective, "top_k_raw_rate")
        self.assertEqual(result.candidate_top_k, 2)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        ordered = np.argsort(-probabilities)[:2]
        top_k_best = max(states[int(index)].sum_rate for index in ordered)
        self.assertGreaterEqual(top_k_best, states[result.top_index].sum_rate - 1e-9)

    def test_scale_challenge_omits_headline_hardware_evidence(self):
        args = Namespace(
            scale_uavs=2,
            scale_grid_points=4,
            scale_survivors=2,
            scale_antennas=2,
            scale_gamma_min=0.0,
            scale_seed=3,
            scale_grid_steps=3,
            scale_shots=16,
            scale_top_k=2,
            scale_random_trials=2,
            scale_sweep_top_k="1,2",
            scale_sweep_random_trials=2,
            scale_sa_restarts=2,
            scale_sa_steps=4,
            scale_sa_trials=2,
            scale_sa_start_temperature=0.25,
            scale_sa_end_temperature=0.01,
            noise_levels="0,1",
            verbose=False,
        )

        result = run_scale_challenge(args)

        self.assertNotIn("hardware_evidence", result)

    def test_suite_reports_candidate_aware_qaoa(self):
        args = Namespace(
            suite_seed_range="1:2",
            suite_uavs=2,
            suite_grid_points=4,
            suite_survivors=2,
            suite_antennas=2,
            suite_gamma_min=0.0,
            suite_grid_steps=3,
            suite_shots=16,
            suite_top_k=2,
            suite_random_trials=2,
            suite_sweep_top_k="1,2",
            suite_sweep_random_trials=2,
            suite_sa_restarts=2,
            suite_sa_steps=4,
            suite_sa_trials=2,
            suite_sa_start_temperature=0.25,
            suite_sa_end_temperature=0.01,
            suite_stress_gap=0.0,
        )

        result = run_suite(args)

        self.assertGreaterEqual(result["all"]["count"], 1)
        self.assertIn("mean_candidate_aware_top_k_local_AR_rate", result["all"])
        self.assertIn("candidate_aware_top_k_sweep", result)
        self.assertEqual(
            len(result["candidate_aware_top_k_sweep"]),
            len(result["top_k_sweep"]),
        )
        for row in result["rows"]:
            self.assertIn("candidate_aware_top_k_local_AR_rate", row)
            self.assertIn("candidate_aware_top_k_sweep", row)
            self.assertGreaterEqual(row["candidate_aware_top_k_local_AR_rate"], 0.0)
            self.assertLessEqual(row["candidate_aware_top_k_local_AR_rate"], 1.0)

    def test_smaller_hardware_evidence_records_completed_ibm_job(self):
        evidence = build_smaller_hardware_evidence_status()

        self.assertEqual(evidence["job_id"], "d91ttqmu9n7c73ane4jg")
        self.assertEqual(evidence["qubit_count"], 18)
        self.assertEqual(evidence["transpiled_depth"], 882)
        self.assertEqual(evidence["two_qubit_gate_count"], 780)
        self.assertEqual(evidence["best_projected_count"], 203)
        self.assertGreater(evidence["projected_optimum_count_lift_vs_random_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
