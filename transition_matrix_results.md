# Transition Matrix Results

Run setting:

- Command style: `python3 policy_solver.py --N <N> --num-transmissions 1000000`
- Simulation transmissions: `1,000,000`
- State notation: `(current_symbol, previous_length)`
- Codebook notation: `[l1, l2, ..., lN]`, where `li` is the codeword length assigned to symbol `i`
- Index: `Optimal inverse-length gain over myopic`
- Index formula: `((1 / optimal_analytic) - (1 / myopic_analytic)) / (1 / myopic_analytic) * 100%`

## Case 2: 5x5 uniform matrix

Matrix used:

```text
0.2 0.2 0.2 0.2 0.2
0.2 0.2 0.2 0.2 0.2
0.2 0.2 0.2 0.2 0.2
0.2 0.2 0.2 0.2 0.2
0.2 0.2 0.2 0.2 0.2
```

Results:

```text
Policy                 Analytic        Simulation
Steady-state           2.400000       2.400103
Myopic                 2.400000       2.400368
Optimal                2.400000       2.399807

Optimal inverse-length gain over myopic: -0.0000%
```

Steady-state policy:

```text
Single codebook for all states: [3, 3, 2, 2, 2]
```

Myopic policy:

```text
Single codebook for all states: [3, 3, 2, 2, 2]
```

Optimal policy:

```text
State-dependent codebooks:
lengths=[2, 2, 3, 3, 2] <- states: (1,1), (1,4), (2,1), (2,4), (3,1), (3,4), (4,1), (4,4), (5,1), (5,4)
lengths=[2, 2, 2, 3, 3] <- states: (1,2), (1,3), (2,2), (2,3), (3,2), (3,3), (4,2), (4,3), (5,2), (5,3)
```

## Case 3: 6x6 two-cluster matrix A

Matrix used:

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
Policy                 Analytic        Simulation
Steady-state           2.424528       2.422909
Myopic                 2.152026       2.153468
Optimal                2.137519       2.137523

Optimal inverse-length gain over myopic: 0.6787%
```

Steady-state policy:

```text
Single codebook for all states: [3, 3, 3, 3, 2, 2]
```

Myopic policy:

```text
State-dependent codebooks:
lengths=[3, 2, 2, 2, 4, 4] <- states: (1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2)
lengths=[3, 3, 2, 2, 3, 3] <- states: (1,3), (1,4), (2,3), (2,4), (3,3), (3,4), (4,3), (4,4)
lengths=[3, 3, 3, 2, 3, 2] <- states: (1,5), (2,5), (3,5), (4,5)
lengths=[5, 5, 4, 3, 2, 1] <- states: (5,1), (6,1)
lengths=[4, 4, 4, 4, 2, 1] <- states: (5,2), (5,3), (5,4), (5,5), (6,2), (6,3), (6,4), (6,5)
```

Optimal policy:

```text
State-dependent codebooks:
lengths=[3, 2, 2, 2, 4, 4] <- states: (1,1), (2,1), (3,1), (4,1)
lengths=[3, 3, 2, 2, 3, 3] <- states: (1,2), (1,3), (2,2), (2,3), (3,2), (3,3), (4,2), (4,3)
lengths=[3, 3, 3, 3, 2, 2] <- states: (1,4), (1,5), (2,4), (2,5), (3,4), (3,5), (4,4), (4,5)
lengths=[5, 5, 4, 3, 2, 1] <- states: (5,1), (6,1)
lengths=[4, 4, 4, 4, 2, 1] <- states: (5,2), (5,3), (5,4), (5,5), (6,2), (6,3), (6,4), (6,5)
```

## Case 4: 6x6 two-cluster matrix B

Note: rows 5 and 6 in the original input sum to `1.009`, so the solver normalized those rows. The matrix below is the row-normalized matrix actually used.

Matrix used:

```text
0.24975 0.24975 0.24975 0.24975 0.0005 0.0005
0.24975 0.24975 0.24975 0.24975 0.0005 0.0005
0.24975 0.24975 0.24975 0.24975 0.0005 0.0005
0.24975 0.24975 0.24975 0.24975 0.0005 0.0005
0.0024777 0.0024777 0.0024777 0.0024777 0.495045 0.495045
0.0024777 0.0024777 0.0024777 0.0024777 0.495045 0.495045
```

Results:

```text
Policy                 Analytic        Simulation
Steady-state           2.344801       2.341013
Myopic                 2.161938       2.164541
Optimal                2.161938       2.160983

Optimal inverse-length gain over myopic: 0.0000%
```

Steady-state policy:

```text
Single codebook for all states: [3, 2, 2, 2, 4, 4]
```

Myopic policy:

```text
State-dependent codebooks:
lengths=[3, 2, 2, 2, 4, 4] <- states: (1,1), (1,2), (1,3), (1,4), (1,5), (2,1), (2,2), (2,3), (2,4), (2,5), (3,1), (3,2), (3,3), (3,4), (3,5), (4,1), (4,2), (4,3), (4,4), (4,5)
lengths=[4, 4, 4, 4, 2, 1] <- states: (5,1), (5,2), (5,3), (5,4), (5,5), (6,1), (6,2), (6,3), (6,4), (6,5)
```

Optimal policy:

```text
State-dependent codebooks:
lengths=[2, 2, 3, 2, 4, 4] <- states: (1,1), (2,1), (3,1), (4,1)
lengths=[2, 2, 2, 3, 4, 4] <- states: (1,2), (1,5), (2,2), (2,5), (3,2), (3,5), (4,2), (4,5)
lengths=[2, 3, 2, 2, 4, 4] <- states: (1,3), (1,4), (2,3), (2,4), (3,3), (3,4), (4,3), (4,4)
lengths=[4, 4, 4, 4, 1, 2] <- states: (5,1), (5,2), (5,3), (5,4), (5,5), (6,1), (6,2), (6,3), (6,4), (6,5)
```
