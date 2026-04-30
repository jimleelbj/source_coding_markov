import argparse
import re
from collections import defaultdict

import numpy as np

import experiment1 as exp1


def parse_row(row_text: str, n: int) -> list[float]:
    tokens = [tok for tok in re.split(r"[,\s]+", row_text.strip()) if tok]
    if len(tokens) != n:
        raise ValueError(f"Expected {n} values, got {len(tokens)}.")
    return [float(tok) for tok in tokens]


def prompt_transition_matrix(n: int) -> np.ndarray:
    print(f"\nPlease input a {n}x{n} transition matrix.")
    print("Input one row at a time, separated by spaces or commas.")
    rows = []
    for i in range(n):
        row_text = input(f"Row {i + 1}: ")
        rows.append(parse_row(row_text, n))
    p = np.array(rows, dtype=float)
    return p


def validate_transition_matrix(p: np.ndarray) -> np.ndarray:
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("Transition matrix must be square (N x N).")
    if np.any(p < 0):
        raise ValueError("Transition matrix cannot contain negative values.")

    row_sums = p.sum(axis=1)
    if np.any(np.isclose(row_sums, 0.0)):
        raise ValueError("Each row must have positive sum.")

    # Normalize each row to avoid tiny user input rounding errors.
    p = p / row_sums[:, None]
    return p


def policy_to_codebook_map(
    policy: list[int],
    all_actions: list[list[int]],
    n: int,
) -> dict[tuple[int, int], list[int]]:
    mapping: dict[tuple[int, int], list[int]] = {}
    for s_idx, action_idx in enumerate(policy):
        symbol, length = exp1.state_from_index(s_idx, n)
        mapping[(symbol, length)] = all_actions[action_idx]
    return mapping


def print_policy_codebooks(
    name: str,
    policy: list[int],
    all_actions: list[list[int]],
    n: int,
) -> None:
    print(f"\n{name} code lengths")
    unique_actions = sorted(set(policy))

    if len(unique_actions) == 1:
        lengths = all_actions[unique_actions[0]]
        print(f"  Single codebook for all states: {lengths}")
        return

    print("  State-dependent codebooks (state = (current_symbol, previous_length)):")
    mapping = policy_to_codebook_map(policy, all_actions, n)
    grouped: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
    for state, lengths in mapping.items():
        grouped[tuple(lengths)].append(state)

    for lengths, states in grouped.items():
        preview = ", ".join(f"({s},{l})" for s, l in states[:8])
        suffix = "" if len(states) <= 8 else f", ... total {len(states)} states"
        print(f"  lengths={list(lengths)} <- states: {preview}{suffix}")


def run_solver(n: int, p: np.ndarray, num_transmissions: int, seed: int) -> None:
    p = validate_transition_matrix(p)
    all_actions = exp1.all_complete_codes(n)
    p_pows = exp1.compute_powers(p, n)
    transition_actions, costs_actions = exp1.build_action_matrices(p_pows, all_actions, n)

    steady_policy = exp1.steady_state_policy(p, n, all_actions)
    myopic_policy = exp1.myopic_policy(p_pows, n, all_actions)
    optimal_policy, optimal_analytic = exp1.relative_value_iteration(transition_actions, costs_actions)

    myopic_analytic, _, _, _ = exp1.evaluate_policy(myopic_policy, transition_actions, costs_actions)
    steady_analytic, _, _, _ = exp1.evaluate_policy(steady_policy, transition_actions, costs_actions)

    optimal_sim = exp1.simulate_policy(
        optimal_policy,
        p_pows,
        all_actions,
        n,
        num_transmissions=num_transmissions,
        seed=seed,
    )
    myopic_sim = exp1.simulate_policy(
        myopic_policy,
        p_pows,
        all_actions,
        n,
        num_transmissions=num_transmissions,
        seed=seed + 1,
    )
    steady_sim = exp1.simulate_policy(
        steady_policy,
        p_pows,
        all_actions,
        n,
        num_transmissions=num_transmissions,
        seed=seed + 2,
    )

    print("\n=== Transition matrix (row-normalized) ===")
    np.set_printoptions(precision=6, suppress=True)
    print(p)

    print_policy_codebooks("Steady-state policy", steady_policy, all_actions, n)
    print_policy_codebooks("Myopic policy", myopic_policy, all_actions, n)
    print_policy_codebooks("Optimal policy", optimal_policy, all_actions, n)

    print("\n=== Average transmission duration ===")
    print("Policy                 Analytic        Simulation")
    print(f"Steady-state       {steady_analytic:12.6f}   {steady_sim:12.6f}")
    print(f"Myopic             {myopic_analytic:12.6f}   {myopic_sim:12.6f}")
    print(f"Optimal            {optimal_analytic:12.6f}   {optimal_sim:12.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute three policies (steady-state, myopic, optimal) for a given Markov transition matrix."
    )
    parser.add_argument("--N", type=int, help="Alphabet size N. If omitted, you will be prompted.")
    parser.add_argument(
        "--num-transmissions",
        type=int,
        default=1_000_000,
        help="Simulation transmissions (default: 1000000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed for simulation.")
    args = parser.parse_args()

    n = args.N if args.N is not None else int(input("Alphabet size N: ").strip())
    if n < 2:
        raise ValueError("N must be at least 2.")

    p = prompt_transition_matrix(n)
    run_solver(n, p, args.num_transmissions, args.seed)


if __name__ == "__main__":
    main()

#python policy_solver.py --N 4 --num-transmissions 1000000
#Please input a 4x4 transition matrix.
#Input one row at a time, separated by spaces or commas.
#Row 1: 0.55 0.25 0.10 0.10
#Row 2: 0.20 0.50 0.20 0.10
#Row 3: 0.15 0.20 0.50 0.15
#Row 4: 0.10 0.15 0.25 0.50
