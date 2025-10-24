import pandas as pd
import numpy as np
from typing import Dict
from plotnine import *

# Board states
X = 1
O = -1
E = 0


def legal_moves(board):
    return board == E

def is_terminal(board):
    results = np.zeros(8)
    results[0:3] = board.sum(axis=0)
    results[3:6] = board.sum(axis=1)
    results[6]   = np.trace(board)
    results[7]   = board[2,0] + board[1,1] + board[0,2]

    if np.any(results == 3):
        return True, X
    if np.any(results == -3):
        return True, O
    if not (board == E).any():
        return True, 0
    return False, None

def apply_move(board, action_idx, player):
    r, c = divmod(action_idx, 3)
    nxt = board.copy()
    nxt[r,c] = player
    return nxt

def base3_key(board):
    key = 0
    for val in board.flatten():
        key = key * 3 + (val + 1)
    return int(key)

class QEntry:
    def __init__(self, board):
        self.board = board.copy()
        self.q = np.full((3,3), np.nan, dtype=float)
        self.q[legal_moves(board)] = 0.5    # Optimistic init

QTable = Dict[int, QEntry]

def ensure_QEntry(Q, board):
    k = base3_key(board)
    if k not in Q:
        Q[k] = QEntry(board)
    return Q[k]

def choose_action(Q, board, epsilon, rng):
    entry = ensure_QEntry(Q, board)
    legal_idxs = np.flatnonzero(legal_moves(board))

    if rng.random() < epsilon:
        return rng.choice(legal_idxs)
    
    qvals = entry.q.flatten()
    legal_qvals = qvals[legal_idxs]
    best_val = legal_qvals.max()
    best_choices = legal_idxs[legal_qvals == best_val]
    return rng.choice(best_choices)

def q_update(Q_self, Q_opp, player, s_board, action_idx, alpha, gamma, epsilon, rng):
    # Ensure current state exists and fetch Q(s,a)
    entry = ensure_QEntry(Q_self, s_board)
    r, c = divmod(action_idx, 3)
    cur_q = entry.q[r, c]
    

    # Apply players move
    s_after = apply_move(s_board, action_idx, player)
    term, result = is_terminal(s_after)

    # Reward from player perspective
    def R(res):
        if res == player:
            return 1.0
        elif res == 0:
            return 0.0
        else:
            return -1.0

    # Terminal on players move => immediate target
    if term:
        target = R(result)
        entry.q[r, c] = cur_q + alpha * (target - cur_q)
        return

    # Opponent responds ε-greedily
    opp_a = choose_action(Q_opp, s_after, epsilon, rng)
    s_after_opp = apply_move(s_after, opp_a, -player)
    term2, result2 = is_terminal(s_after_opp)

    if term2:
        target = R(result2)
    else:
        # Its players turn again. bootstrap on players next-turn state
        entry_next = ensure_QEntry(Q_self, s_after_opp)
        max_next = np.nanmax(entry_next.q)
        if np.isnan(max_next):  # (defensive; shouldn't happen with legal moves)            ########## REDUNDAND
            max_next = 0.0
        target = gamma * max_next


    entry.q[r, c] = cur_q + alpha * (target - cur_q)

  
##### TRAINING #####

def train(
        rounds=200_000,
        alpha=0.2,
        gamma=1.0,
        seed=42,
):
    rng = np.random.default_rng(seed)

    Q_x: QTable = {}
    Q_o: QTable = {}

    outcomes = []

    for round in range(1, rounds + 1):
        board = np.zeros((3, 3), dtype=int)
        current_player = X

        if round <= 10**4:
            epsilon = 1.0
        else:
            decay_steps = (round - 10**4) // 100
            epsilon = 0.9 ** decay_steps


        while True:
            Qcur = Q_x if current_player == X else Q_o

            # Choose action and update current players Q trough other players Q
            a = choose_action(Qcur, board, epsilon, rng)
            if current_player == X:
                q_update(Q_x, Q_o, X, board, a, alpha, gamma, epsilon, rng)
            else:
                q_update(Q_o, Q_x, O, board, a, alpha, gamma, epsilon, rng)

            # Results of choosen action
            board_after = apply_move(board, a, current_player)
            terminal, result = is_terminal(board_after)

            if terminal:
                outcomes.append(result)
                break

            # give board to next player
            board = board_after
            current_player = -current_player

    return Q_x, Q_o, np.array(outcomes, dtype=int)


###### CSV EXPORT ######

def export_qtable_to_csv(Q, filepath):
    boards = []
    qvals  = []
    for k in sorted(Q):
        entry = Q[k]
        board = entry.board.astype(float)
        q = entry.q.copy()
        q[board != E] = np.nan
        boards.append(board)
        qvals.append(q)

    out = np.vstack([
        np.hstack(boards),
        np.hstack(qvals)
    ])
    np.savetxt(filepath, out, delimiter=",", fmt="%.8f")

###### PLOTTING #####

def plot_window(outcomes, W, fname="training_fractions_window.png"):
    out = np.asarray(outcomes)
    rounds = np.arange(1, out.size + 1)

    df_ep = pd.DataFrame({
        "round": rounds,
        "P1 (X) wins": (out == X).astype(float),
        "P2 (O) wins": (out == O).astype(float),
        "Draws":       (out == 0).astype(float),
    })

    minp = W if out.size >= W else 1
    df_roll = pd.DataFrame({
        "round": df_ep["round"],
        "P1 (X) wins": df_ep["P1 (X) wins"].rolling(W, min_periods=minp).mean(),
        "P2 (O) wins": df_ep["P2 (O) wins"].rolling(W, min_periods=minp).mean(),
        "Draws":        df_ep["Draws"].rolling(W, min_periods=minp).mean(),
    })

    df_long = df_roll.melt(id_vars="round", var_name="Outcome", value_name="Fraction").dropna(subset=["Fraction"])

    p = (
        ggplot(df_long, aes("round", "Fraction", color="Outcome"))
        + geom_line()
        + labs(
            x="round",
            y="Frequency",
            color="Outcome"
        )
        + theme_minimal()
        + guides(color=guide_legend(title=""))
    )

    print(p)
    p.save(fname, dpi=150, width=9, height=5)



##### MAIN CODE #####

if __name__ == "__main__":

    Q_player1, Q_player2, outcomes = train(
        rounds=100_000,
        alpha=0.1,
        gamma=1.0,
        seed=42,
    )

    plot_window(outcomes, W=2000)

    export_qtable_to_csv(Q_player1, "player1.csv")
    export_qtable_to_csv(Q_player2, "player2.csv")

