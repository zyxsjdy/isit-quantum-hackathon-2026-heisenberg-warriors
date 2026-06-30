import unittest

import numpy as np

from qaoa_isac_benchmark import (
    build_full_binary_qaoa_circuit,
    build_environment,
    build_probability_noise_robustness,
    compare_qaoa_vs_random_local_search,
    circuit_summary,
    enumerate_assignments,
    extract_sampler_counts,
    parse_float_list,
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


if __name__ == "__main__":
    unittest.main()
