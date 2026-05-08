# Codex Handoff Notes

This file records the working context from the Codex chat so another machine/session can continue without losing the thread.

## Project Goal

We are reproducing and exploring the paper:

`Optimal Source Coding of Markov Chains for Real-Time Remote Estimation`

The main object is a state-dependent source-coding policy:

```text
phi : S -> U_N
S = {(current_symbol, previous_length)}
```

Important conceptual point:

- A policy is one fixed mapping from every state to a codebook.
- A state-dependent policy may use different codebooks in different states.
- The analytic/simulation value used for comparison is the full policy value, not the value of any one codebook forced onto every state.

## Important Formula / Index

The current custom index is:

```text
Optimal inverse-length gain over myopic
= ((1 / optimal_analytic) - (1 / myopic_analytic)) / (1 / myopic_analytic) * 100%
```

Equivalent simplification:

```text
(myopic_analytic / optimal_analytic - 1) * 100%
```

Interpretation:

- Positive means optimal has a higher inverse average length, i.e. better throughput-like value.
- If comparing raw average length instead, improvement would appear as a negative change because lower average length is better.

## Code State

Main files touched:

- `policy_solver.py`
- `experiment1.py`
- `transition_matrix_results.md`
- `CODEX_HANDOFF.md`

### `policy_solver.py`

Current behavior:

- Prompts for an `N x N` transition matrix.
- Row-normalizes the matrix before solving.
- Prints:
  - row-normalized transition matrix
  - steady-state policy code lengths
  - myopic policy code lengths
  - optimal policy code lengths
  - analytic and simulation average transmission duration
  - optimal inverse-length gain over myopic

Important output convention:

```text
lengths=[...] <- states: (...)
```

This means those states use that codebook inside the full policy.
It does not mean that codebook alone is being simulated as a full policy.

Current default in `policy_solver.py` may be `1_000_000` transmissions. Verify line around parser argument:

```python
parser.add_argument(
    "--num-transmissions",
    type=int,
    default=1_000_000,
    help="Simulation transmissions (default: 1000000).",
)
```

Going forward, use `--num-transmissions 1000000` explicitly for consistency.

### `experiment1.py`

`matplotlib.pyplot` was moved inside `main()` so importing `experiment1` from `policy_solver.py` does not require matplotlib unless plotting experiment 1.

`relative_value_iteration_details(...)` was added as a helper, but `policy_solver.py` currently uses the normal:

```python
exp1.relative_value_iteration(...)
```

## Important Clarifications From Discussion

### What is eta in equation 14?

`eta` is the long-term average cost, i.e. average transmission duration under a policy.

In average-cost MDP Bellman equations:

```text
V_s = min_u { c(s,u) - eta + sum_s' T(s,s',u)V_s' }
```

`V_s` is a relative/bias value. There are `|S|` values plus `eta`, so `|S| + 1` unknowns. One state's value is fixed, e.g. `V_(1,1)=0`, because relative values are only defined up to a constant shift.

### What is the optimal policy?

The target is:

```text
phi*: S -> U_N
```

The solver finds a codebook for every state. The full mapping is the policy. During simulation, each step:

1. observes current state `(symbol, previous_length)`
2. looks up the codebook assigned by the policy
3. samples the next symbol according to `P^previous_length`
4. uses the codeword length assigned to that next symbol
5. moves to the next augmented state

### Multiple optimal policies

If multiple optimal policies exist, their analytic average costs are the same in theory.

The implementation uses `np.argmin`, so when actions tie, it chooses the first tied action index. Simulation uses only that selected full policy.

We briefly experimented with printing/simulating tied actions and single-codebook checks, but those were removed because they distracted from the paper's actual full-policy comparison.

## Recorded Results

Detailed results are in:

`transition_matrix_results.md`

That file includes cases 2, 3, and 4, all rerun with:

```text
num_transmissions = 1,000,000
```

It records:

- matrix used
- analytic/simulation values
- inverse-length gain index
- steady-state/myopic/optimal policy codebook assignments

### Case 2: 5x5 uniform matrix

Matrix:

```text
0.2 0.2 0.2 0.2 0.2
0.2 0.2 0.2 0.2 0.2
0.2 0.2 0.2 0.2 0.2
0.2 0.2 0.2 0.2 0.2
0.2 0.2 0.2 0.2 0.2
```

Results:

```text
Steady-state analytic=2.400000 simulation=2.400103
Myopic       analytic=2.400000 simulation=2.400368
Optimal      analytic=2.400000 simulation=2.399807
Gain=-0.0000%
```

### Case 3: 6x6 two-cluster matrix A

Matrix:

```text
0.21 0.22 0.23 0.24 0.06 0.04
0.21 0.22 0.23 0.24 0.06 0.04
0.21 0.22 0.23 0.24 0.06 0.04
0.21 0.22 0.23 0.24 0.06 0.04
0.01 0.02 0.03 0.04 0.41 0.49
0.01 0.02 0.03 0.04 0.41 0.49
```

Results:

```text
Steady-state analytic=2.424528 simulation=2.422909
Myopic       analytic=2.152026 simulation=2.153468
Optimal      analytic=2.137519 simulation=2.137523
Gain=0.6787%
```

### Case 4: 6x6 two-cluster matrix B

Original rows 5 and 6 summed to `1.009`, so solver normalized them. See `transition_matrix_results.md` for the actual normalized matrix.

Results:

```text
Steady-state analytic=2.344801 simulation=2.341013
Myopic       analytic=2.161938 simulation=2.164541
Optimal      analytic=2.161938 simulation=2.160983
Gain=0.0000%
```

## Other Matrices Discussed

### Initial 4x4 debug/example matrix

```text
0.55 0.25 0.10 0.10
0.20 0.50 0.20 0.10
0.15 0.20 0.50 0.15
0.10 0.15 0.25 0.50
```

### 8x8 grouped matrix

Groups:

- states 1-4
- states 5-7
- state 8 alone

Matrix:

```text
0.238 0.252 0.224 0.236 0.012 0.010 0.008 0.020
0.230 0.246 0.244 0.230 0.010 0.012 0.008 0.020
0.242 0.226 0.248 0.234 0.008 0.012 0.010 0.020
0.236 0.234 0.240 0.240 0.010 0.008 0.012 0.020
0.012 0.010 0.008 0.020 0.306 0.324 0.300 0.020
0.010 0.012 0.008 0.020 0.318 0.302 0.310 0.020
0.008 0.010 0.012 0.020 0.296 0.326 0.308 0.020
0.018 0.014 0.010 0.018 0.012 0.014 0.014 0.900
```

It was run with `10,000` transmissions, not the later standard `1,000,000`.

Results from that run:

```text
Steady-state analytic=3.000000 simulation=3.000000
Myopic       analytic=2.206600 simulation=2.200400
Optimal      analytic=2.193400 simulation=2.183900
Gain=0.6018%
```

If this case is needed for final records, rerun with `--num-transmissions 1000000`.

## Environment / Dependency Notes

Initial issue:

- `numpy` was missing from `/home/jimleelbj/miniforge3/bin/python3`

Installed:

- `numpy 2.4.4`
- `matplotlib 3.10.9`

The import dependency was cleaned up by moving matplotlib import inside `experiment1.main()`.

## Suggested Next Steps

On the next machine/session:

1. Open `CODEX_HANDOFF.md`.
2. Open `transition_matrix_results.md`.
3. Check `policy_solver.py` default `--num-transmissions`.
4. If producing final experimental numbers, run all selected matrices with:

```bash
python3 policy_solver.py --N <N> --num-transmissions 1000000
```

5. Use analytic values for policy comparison; simulation is only a Monte Carlo check.
