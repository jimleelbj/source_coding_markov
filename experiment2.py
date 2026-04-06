import argparse
import numpy as np
import matplotlib.pyplot as plt

import experiment1 as exp1


def random_transition_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.random((n, n))
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def average_costs_for_n(
    n: int,
    betas: np.ndarray,
    num_random_matrices: int,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_actions = exp1.all_complete_codes(n)
    h_alpha = alpha * np.eye(n) + ((1 - alpha) / (n - 1)) * (np.ones((n, n)) - np.eye(n))

    sum_optimal = np.zeros(len(betas))
    sum_myopic = np.zeros(len(betas))
    sum_steady = np.zeros(len(betas))

    for idx in range(num_random_matrices):
        random_matrix = random_transition_matrix(n, rng)
        for beta_idx, beta in enumerate(betas):
            p = (1 - beta) * h_alpha + beta * random_matrix
            p_powers = exp1.compute_powers(p, n)
            transition_actions, costs_actions = exp1.build_action_matrices(p_powers, all_actions, n)

            steady_policy = exp1.steady_state_policy(p, n, all_actions)
            myopic_policy = exp1.myopic_policy(p_powers, n, all_actions)
            _, optimal_cost = exp1.relative_value_iteration(transition_actions, costs_actions)
            myopic_cost, _, _, _ = exp1.evaluate_policy(myopic_policy, transition_actions, costs_actions)
            steady_cost, _, _, _ = exp1.evaluate_policy(steady_policy, transition_actions, costs_actions)

            sum_optimal[beta_idx] += optimal_cost
            sum_myopic[beta_idx] += myopic_cost
            sum_steady[beta_idx] += steady_cost

        if (idx + 1) % 1000 == 0 or idx + 1 == num_random_matrices:
            print(f"N={n}: processed {idx + 1}/{num_random_matrices} random matrices")

    avg_optimal = sum_optimal / num_random_matrices
    avg_myopic = sum_myopic / num_random_matrices
    avg_steady = sum_steady / num_random_matrices
    return avg_optimal, avg_myopic, avg_steady


def plot_single_figure(
    betas: np.ndarray,
    avg_optimal: np.ndarray,
    avg_myopic: np.ndarray,
    avg_steady: np.ndarray,
    n: int,
    output_path: str,
) -> None:
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(betas, avg_steady, lw=2, label="Steady-state Huffman")
    plt.plot(betas, avg_myopic, lw=2, label="Myopic Huffman")
    plt.plot(betas, avg_optimal, lw=2, label="Optimal policy")
    plt.xlabel(r"$\beta$")
    plt.ylabel("Average transmission duration")
    plt.title(f"Figure 5 ({'a' if n == 4 else 'b'}): N={n}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Figure 5 (Experiment 2) from the paper.")
    parser.add_argument("--num-random-matrices", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta-step", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    betas = np.arange(0.0, 1.0 + args.beta_step / 2, args.beta_step)
    print(
        f"Running Experiment 2 with alpha={args.alpha}, betas={len(betas)} points, "
        f"num_random_matrices={args.num_random_matrices}"
    )

    avg_opt_4, avg_myo_4, avg_std_4 = average_costs_for_n(
        n=4,
        betas=betas,
        num_random_matrices=args.num_random_matrices,
        alpha=args.alpha,
        seed=args.seed,
    )
    avg_opt_5, avg_myo_5, avg_std_5 = average_costs_for_n(
        n=5,
        betas=betas,
        num_random_matrices=args.num_random_matrices,
        alpha=args.alpha,
        seed=args.seed + 1,
    )

    plot_single_figure(
        betas=betas,
        avg_optimal=avg_opt_4,
        avg_myopic=avg_myo_4,
        avg_steady=avg_std_4,
        n=4,
        output_path="experiment2_fig5a_n4.png",
    )
    plot_single_figure(
        betas=betas,
        avg_optimal=avg_opt_5,
        avg_myopic=avg_myo_5,
        avg_steady=avg_std_5,
        n=5,
        output_path="experiment2_fig5b_n5.png",
    )

    print("Saved experiment2_fig5a_n4.png")
    print("Saved experiment2_fig5b_n5.png")


if __name__ == "__main__":
    main()
