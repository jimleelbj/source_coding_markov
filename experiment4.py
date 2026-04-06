import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np

import experiment1 as exp1


def random_transition_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.random((n, n))
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def evaluate_for_n(
    n: int,
    num_random_matrices: int,
    seed: int,
    progress_every: int = 500,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    all_actions = exp1.all_complete_codes(n)

    sum_lst = 0.0
    sum_lm = 0.0
    sum_lstar = 0.0
    sum_gain_st = 0.0
    sum_gain_m = 0.0

    start_time = time.time()
    for idx in range(num_random_matrices):
        p = random_transition_matrix(n, rng)
        p_powers = exp1.compute_powers(p, n)
        transition_actions, costs_actions = exp1.build_action_matrices(p_powers, all_actions, n)

        steady_policy = exp1.steady_state_policy(p, n, all_actions)
        myopic_policy = exp1.myopic_policy(p_powers, n, all_actions)
        _, lstar = exp1.relative_value_iteration(transition_actions, costs_actions)
        lm, _, _, _ = exp1.evaluate_policy(myopic_policy, transition_actions, costs_actions)
        lst, _, _, _ = exp1.evaluate_policy(steady_policy, transition_actions, costs_actions)

        sum_lstar += lstar
        sum_lm += lm
        sum_lst += lst
        sum_gain_st += lst - lstar
        sum_gain_m += lm - lstar

        if (idx + 1) % progress_every == 0 or idx + 1 == num_random_matrices:
            elapsed = time.time() - start_time
            print(
                f"N={n}: processed {idx + 1}/{num_random_matrices} "
                f"(elapsed {elapsed:.1f}s)"
            )

    denom = float(num_random_matrices)
    return {
        "E_Lst": sum_lst / denom,
        "E_Lm": sum_lm / denom,
        "E_Lstar": sum_lstar / denom,
        "E_Lst_minus_Lstar": sum_gain_st / denom,
        "E_Lm_minus_Lstar": sum_gain_m / denom,
    }


def build_table_matrix(results: dict[int, dict[str, float]]) -> tuple[list[str], list[str], list[list[str]]]:
    row_labels = [
        "E[Lst]",
        "E[Lm]",
        "E[L*]",
        "E[Lst - L*]",
        "E[Lm - L*]",
    ]
    col_labels = [f"N = {n}" for n in sorted(results)]

    key_order = [
        "E_Lst",
        "E_Lm",
        "E_Lstar",
        "E_Lst_minus_Lstar",
        "E_Lm_minus_Lstar",
    ]

    cell_text = []
    for key in key_order:
        row = []
        for n in sorted(results):
            row.append(f"{results[n][key]:.4f}")
        cell_text.append(row)

    return row_labels, col_labels, cell_text


def print_console_table(results: dict[int, dict[str, float]]) -> None:
    row_labels, col_labels, cell_text = build_table_matrix(results)

    print("\nTABLE II FORMAT (Experiment 4)\n")
    header = "Metric".ljust(16) + " ".join(label.rjust(10) for label in col_labels)
    print(header)
    print("-" * len(header))
    for label, row in zip(row_labels, cell_text):
        print(label.ljust(16) + " ".join(value.rjust(10) for value in row))


def save_csv(results: dict[int, dict[str, float]], path: str) -> None:
    row_labels, col_labels, cell_text = build_table_matrix(results)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + col_labels)
        for label, row in zip(row_labels, cell_text):
            writer.writerow([label] + row)


def save_table_png(results: dict[int, dict[str, float]], path: str) -> None:
    row_labels, col_labels, cell_text = build_table_matrix(results)

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    ax.set_title("Experiment 4 Results (Table II Format)", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Experiment 4 in Table II format.")
    parser.add_argument("--num-random-matrices", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    print(
        f"Running Experiment 4 with num_random_matrices={args.num_random_matrices}, "
        f"seed={args.seed}"
    )

    results = {}
    n_values = [3, 4, 5, 6]
    for i, n in enumerate(n_values):
        results[n] = evaluate_for_n(
            n=n,
            num_random_matrices=args.num_random_matrices,
            seed=args.seed + i,
            progress_every=args.progress_every,
        )

    print_console_table(results)
    save_csv(results, "experiment4_table2.csv")
    save_table_png(results, "experiment4_table2.png")
    print("\nSaved experiment4_table2.csv")
    print("Saved experiment4_table2.png")


if __name__ == "__main__":
    main()
