import heapq
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def all_complete_codes(N: int) -> list[list[int]]:
    codes = []
    max_len = N - 1
    for lengths in np.ndindex(*(max_len,) * N):
        lengths = [l + 1 for l in lengths]
        if abs(sum(2.0 ** (-l) for l in lengths) - 1.0) < 1e-12:
            codes.append(lengths)
    return codes


def state_index(symbol: int, length: int, N: int) -> int:
    return (symbol - 1) * (N - 1) + (length - 1)


def state_from_index(index: int, N: int) -> tuple[int, int]:
    symbol = index // (N - 1) + 1
    length = index % (N - 1) + 1
    return symbol, length


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    S = T.shape[0]
    A = T.T - np.eye(S)
    A[-1, :] = 1.0
    b = np.zeros(S)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def evaluate_policy(policy: list[int], T_actions: list[np.ndarray], c_actions: list[np.ndarray]):
    S = len(policy)
    T = np.zeros((S, S))
    c = np.zeros(S)
    for s in range(S):
        T[s, :] = T_actions[policy[s]][s, :]
        c[s] = c_actions[policy[s]][s]
    pi = stationary_distribution(T)
    J = float(np.dot(pi, c))
    A = np.eye(S) - T
    b = c - J
    A[0, :] = 1.0
    b[0] = 0.0
    try:
        V = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        V, *_ = np.linalg.lstsq(A, b, rcond=None)
    return J, V, pi, T


def relative_value_iteration(
    T_actions: list[np.ndarray],
    c_actions: list[np.ndarray],
    tolerance: float = 1e-11,
    max_iterations: int = 20000,
):
    """Solve average-cost MDP via relative value iteration.

    This avoids policy-oscillation issues that may appear with direct
    policy-iteration updates on numerically close actions.
    """
    num_states = c_actions[0].shape[0]
    num_actions = len(T_actions)
    bias = np.zeros(num_states)
    reference_state = 0

    for _ in range(max_iterations):
        q_values = np.empty((num_actions, num_states))
        for action_idx in range(num_actions):
            q_values[action_idx] = c_actions[action_idx] + T_actions[action_idx] @ bias

        updated_bias = np.min(q_values, axis=0)
        updated_bias -= updated_bias[reference_state]

        if np.max(np.abs(updated_bias - bias)) < tolerance:
            bias = updated_bias
            break
        bias = updated_bias

    q_values = np.empty((num_actions, num_states))
    for action_idx in range(num_actions):
        q_values[action_idx] = c_actions[action_idx] + T_actions[action_idx] @ bias
    optimal_policy = list(np.argmin(q_values, axis=0))
    optimal_cost, _, _, _ = evaluate_policy(optimal_policy, T_actions, c_actions)
    return optimal_policy, optimal_cost


def compute_powers(P: np.ndarray, N: int) -> list[np.ndarray]:
    return [np.linalg.matrix_power(P, l) for l in range(N)]


def build_action_matrices(P_pows: list[np.ndarray], all_actions: list[list[int]], N: int):
    S = N * (N - 1)
    T_actions = []
    c_actions = []
    for action in all_actions:
        T = np.zeros((S, S))
        c = np.zeros(S)
        for s_idx in range(S):
            symbol, length = state_from_index(s_idx, N)
            row = P_pows[length][symbol - 1]
            c[s_idx] = float(np.dot(row, action))
            for next_symbol in range(1, N + 1):
                next_length = action[next_symbol - 1]
                next_index = state_index(next_symbol, next_length, N)
                T[s_idx, next_index] += row[next_symbol - 1]
        T_actions.append(T)
        c_actions.append(c)
    return T_actions, c_actions


def compute_steady_state_distribution(P: np.ndarray) -> np.ndarray:
    N = P.shape[0]
    A = P.T - np.eye(N)
    A[-1, :] = 1.0
    b = np.zeros(N)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def simulate_policy(policy: list[int], P_pows: list[np.ndarray], all_actions: list[list[int]], N: int, num_transmissions: int = 5000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    cum_probs = [np.cumsum(P_pows[l], axis=1) for l in range(len(P_pows))]
    symbol = 1
    length = 1
    total_time = 0
    for _ in range(num_transmissions):
        row = cum_probs[length][symbol - 1]
        r = rng.random()
        next_symbol = int(np.searchsorted(row, r)) + 1
        action = all_actions[policy[state_index(symbol, length, N)]]
        next_length = action[next_symbol - 1]
        total_time += next_length
        symbol, length = next_symbol, next_length
    return total_time / num_transmissions


def huffman_lengths(p: list[float]) -> list[int]:
    if len(p) < 2:
        raise ValueError("Need at least 2 symbols.")
    heap = [(float(w), i, (i,)) for i, w in enumerate(p)]
    heapq.heapify(heap)
    lengths = [0] * len(p)
    uid = len(p)
    while len(heap) > 1:
        w1, _, s1 = heapq.heappop(heap)
        w2, _, s2 = heapq.heappop(heap)
        for i in s1:
            lengths[i] += 1
        for i in s2:
            lengths[i] += 1
        heapq.heappush(heap, (w1 + w2, uid, s1 + s2))
        uid += 1
    return lengths


def myopic_policy(P_pows: list[np.ndarray], N: int, all_actions: list[list[int]]) -> list[int]:
    policy = []
    for s_idx in range(N * (N - 1)):
        symbol, length = state_from_index(s_idx, N)
        probs = P_pows[length][symbol - 1]
        lengths = huffman_lengths(list(probs))
        policy.append(all_actions.index(lengths))
    return policy


def steady_state_policy(P: np.ndarray, N: int, all_actions: list[list[int]]) -> list[int]:
    pi = compute_steady_state_distribution(P)
    lengths = huffman_lengths(list(pi))
    action_index = all_actions.index(lengths)
    return [action_index] * (N * (N - 1))


def main():
    N = 4
    alpha = 0.5
    num_transmissions = 1_000_000
    R0 = np.array([
        [0.1426, 0.4996, 0.0409, 0.3169],
        [0.3542, 0.5398, 0.0858, 0.0202],
        [0.1732, 0.3522, 0.0946, 0.3800],
        [0.1124, 0.3401, 0.2936, 0.2540],
    ])

    all_actions = all_complete_codes(N)
    optimal_values = []
    myopic_values = []
    steady_values = []
    optimal_sim = []
    myopic_sim = []
    steady_sim = []
    betas = np.arange(0.0, 1.0001, 0.05)

    for beta in betas:
        H_alpha = alpha * np.eye(N) + ((1 - alpha) / (N - 1)) * (np.ones((N, N)) - np.eye(N))
        P = (1 - beta) * H_alpha + beta * R0
        P = P / P.sum(axis=1, keepdims=True)
        P_pows = compute_powers(P, N)
        T_actions, c_actions = build_action_matrices(P_pows, all_actions, N)

        steady_policy_ = steady_state_policy(P, N, all_actions)
        myopic_policy_assign = myopic_policy(P_pows, N, all_actions)
        optimal_policy, optimal_cost = relative_value_iteration(T_actions, c_actions)
        myopic_cost, _, _, _ = evaluate_policy(myopic_policy_assign, T_actions, c_actions)
        steady_cost, _, _, _ = evaluate_policy(steady_policy_, T_actions, c_actions)

        optimal_values.append(optimal_cost)
        myopic_values.append(myopic_cost)
        steady_values.append(steady_cost)
        optimal_sim.append(simulate_policy(optimal_policy, P_pows, all_actions, N, num_transmissions=num_transmissions, seed=42))
        myopic_sim.append(simulate_policy(myopic_policy_assign, P_pows, all_actions, N, num_transmissions=num_transmissions, seed=43))
        steady_sim.append(simulate_policy(steady_policy_, P_pows, all_actions, N, num_transmissions=num_transmissions, seed=44))

    plt.figure(figsize=(10, 6))
    color_optimal = '#1f77b4'
    color_myopic = '#ff7f0e'
    color_steady = '#2ca02c'

    plt.plot(betas, optimal_values, '-', lw=2, color=color_optimal, label='Optimal policy (analytic)')
    plt.plot(betas, myopic_values, '-', lw=2, color=color_myopic, label='Myopic Huffman (analytic)')
    plt.plot(betas, steady_values, '-', lw=2, color=color_steady, label='Steady-state Huffman (analytic)')
    plt.plot(
        betas,
        optimal_sim,
        'o',
        ms=8,
        color=color_optimal,
        label='Optimal policy (simulation)',
    )
    plt.plot(
        betas,
        myopic_sim,
        'o',
        ms=8,
        color=color_myopic,
        label='Myopic Huffman (simulation)',
    )
    plt.plot(
        betas,
        steady_sim,
        'o',
        ms=8,
        color=color_steady,
        label='Steady-state Huffman (simulation)',
    )
    plt.xlabel(r'$\beta$ (mixing weight, with $\alpha = 0.5$ fixed)')
    plt.ylabel('Average transmission duration')
    plt.title('Experiment 1: analytic lines and simulation circles for N=4')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiment1_fig4.png')
    print('saved experiment1_fig4.png')


if __name__ == '__main__':
    main()
