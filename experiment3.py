import argparse
import numpy as np
import matplotlib.pyplot as plt

import experiment1 as exp1


def random_transition_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.random((n, n))
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def cdf_from_samples(samples: np.ndarray, taus: np.ndarray) -> np.ndarray:
    sorted_samples = np.sort(samples)
    counts = np.searchsorted(sorted_samples, taus, side="right")
    return counts / len(sorted_samples)


def evaluate_gains_for_n(
    n: int,
    num_random_matrices: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_actions = exp1.all_complete_codes(n)

    gains_myopic = np.zeros(num_random_matrices)
    gains_steady = np.zeros(num_random_matrices)

    for idx in range(num_random_matrices):
        p = random_transition_matrix(n, rng)
        p_powers = exp1.compute_powers(p, n)
        transition_actions, costs_actions = exp1.build_action_matrices(p_powers, all_actions, n)

        steady_policy = exp1.steady_state_policy(p, n, all_actions)
        myopic_policy = exp1.myopic_policy(p_powers, n, all_actions)
        _, optimal_cost = exp1.relative_value_iteration(transition_actions, costs_actions)
        myopic_cost, _, _, _ = exp1.evaluate_policy(myopic_policy, transition_actions, costs_actions)
        steady_cost, _, _, _ = exp1.evaluate_policy(steady_policy, transition_actions, costs_actions)

        gains_myopic[idx] = myopic_cost - optimal_cost
        gains_steady[idx] = steady_cost - optimal_cost

        if (idx + 1) % 1000 == 0 or idx + 1 == num_random_matrices:
            print(f"N={n}: processed {idx + 1}/{num_random_matrices} random matrices")

    return gains_myopic, gains_steady


def build_tau_grid(gains_myopic: np.ndarray, gains_steady: np.ndarray, num_points: int) -> np.ndarray:
    tau_max = max(float(np.max(gains_myopic)), float(np.max(gains_steady)))
    tau_max = max(0.05, tau_max * 1.05)
    return np.linspace(0.0, tau_max, num_points)


def plot_single_subplot(
    ax: plt.Axes,
    n: int,
    gains_myopic: np.ndarray,
    gains_steady: np.ndarray,
    num_tau_points: int,
) -> None:
    taus = build_tau_grid(gains_myopic, gains_steady, num_tau_points)
    prob_myopic = cdf_from_samples(gains_myopic, taus)
    prob_steady = cdf_from_samples(gains_steady, taus)

    ax.plot(taus, prob_myopic, lw=2, label=r"$\Pr(L_m - L^* \leq \tau)$")
    ax.plot(taus, prob_steady, lw=2, label=r"$\Pr(L_{st} - L^* \leq \tau)$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Probability")
    ax.set_title(f"N = {n}")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True)
    ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Figure 6 (Experiment 3) from the paper.")
    parser.add_argument("--num-random-matrices", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-tau-points", type=int, default=400)
    args = parser.parse_args()

    print(
        f"Running Experiment 3 with num_random_matrices={args.num_random_matrices}, "
        f"num_tau_points={args.num_tau_points}"
    )

    gains_m_4, gains_st_4 = evaluate_gains_for_n(
        n=4,
        num_random_matrices=args.num_random_matrices,
        seed=args.seed,
    )
    gains_m_5, gains_st_5 = evaluate_gains_for_n(
        n=5,
        num_random_matrices=args.num_random_matrices,
        seed=args.seed + 1,
    )

    print(f"N=4: Pr(Lm-L*<=0)={np.mean(gains_m_4 <= 0):.4f}, Pr(Lst-L*<=0)={np.mean(gains_st_4 <= 0):.4f}")
    print(f"N=5: Pr(Lm-L*<=0)={np.mean(gains_m_5 <= 0):.4f}, Pr(Lst-L*<=0)={np.mean(gains_st_5 <= 0):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_single_subplot(axes[0], 4, gains_m_4, gains_st_4, args.num_tau_points)
    plot_single_subplot(axes[1], 5, gains_m_5, gains_st_5, args.num_tau_points)
    fig.suptitle("Figure 6: Performance Gain CDF of the Optimal Policy", y=1.02)
    fig.tight_layout()
    fig.savefig("experiment3_fig6.png", dpi=160)
    plt.close(fig)

    print("Saved experiment3_fig6.png")


if __name__ == "__main__":
    main()
