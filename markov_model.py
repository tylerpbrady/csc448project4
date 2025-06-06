import numpy as np
import sklearn
import scipy
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from scipy.special import softmax



def task_3():
    given_states = np.array([
        [0.86, 0.09, 0.01, 0.03, 0.01],
        [0.01, 0.75, 0.07, 0.08, 0.09],
        [0.01, 0.02, 0.74, 0.21, 0.02],
        [0.21, 0.24, 0.22, 0.21, 0.12],
        [0.01, 0.16, 0.05, 0.05, 0.73]
    ])

    state_vector = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    for _ in range(10):
        state_vector = state_vector @ given_states

    print(state_vector[2])

def task_3_pt2():

    given_states = np.array([
        [0.86, 0.09, 0.01, 0.03, 0.01],
        [0.01, 0.75, 0.07, 0.08, 0.09],
        [0.01, 0.02, 0.74, 0.21, 0.02],
        [0.21, 0.24, 0.22, 0.21, 0.12],
        [0.01, 0.16, 0.05, 0.05, 0.73]
    ])

    state_vector = np.array([0.2, 0.2, 0.2, 0.2, 0.2])


    states = ["S1", "S2", "S3", "S4", "S5"]
    n_states = len(states)
    n_steps = 10

    start_prob = np.full(n_states, 1 / n_states)
    viterbi = np.zeros((n_steps + 1, n_states))
    backtrack = np.zeros((n_steps + 1, n_states), dtype=int)


    viterbi[1] = start_prob


    for t in range(2, n_steps + 1):
        for curr in range(n_states):
            max_prob = 0
            best_prev = 0
            for prev in range(n_states):
                prob = viterbi[t - 1][prev] * given_states[prev][curr]
                if prob > max_prob:
                    max_prob = prob
                    best_prev = prev
            viterbi[t][curr] = max_prob
            backtrack[t][curr] = best_prev

    path_indices = [2]
    for t in reversed(range(2, n_steps + 1)):
        prev = backtrack[t][path_indices[0]]
        path_indices.insert(0, prev)


    most_likely_path = [states[i] for i in path_indices]
    final_probability = viterbi[n_steps][2]

    print(f"Most likely path to S3 at t = 10: {most_likely_path}")
    print(f"Final probability of being in S3: {final_probability}")

def task_4_viterbi_final():
    # Transition probabilities (from state i to state j)
    transition_matrix = np.array([
        [0.86, 0.09, 0.01, 0.03, 0.01],
        [0.01, 0.75, 0.07, 0.08, 0.09],
        [0.01, 0.02, 0.74, 0.21, 0.02],
        [0.21, 0.24, 0.22, 0.21, 0.12],
        [0.01, 0.16, 0.05, 0.05, 0.73]
    ])

    states = ["S1", "S2", "S3", "S4", "S5"]
    num_states = len(states)

    # Load data again
    states_df = pd.read_csv("genemarkers_states.tsv", sep="\t", index_col=0)
    timepoints_df = pd.read_csv("genemarkers_timepoints.tsv", sep="\t", index_col=0)

    # Compute emission probabilities using cosine distance


    dist_matrix = cosine_distances(timepoints_df.values, states_df.values)
    emission_probs = softmax(-dist_matrix, axis=1)  # Shape: [T x S]


    states = ["S1", "S2", "S3", "S4", "S5"]
    timepoints = timepoints_df.index

    E = pd.DataFrame(emission_probs, index=timepoints, columns=states)
    E.to_csv("emission_matrix.csv")


    num_timepoints = emission_probs.shape[0]

    # Initialize Viterbi tables
    viterbi = np.zeros((num_timepoints, num_states))
    backtrack = np.zeros((num_timepoints, num_states), dtype=int)
    start_prob = np.full(num_states, 1 / num_states)

    # Base case: t=0
    for s in range(num_states):
        viterbi[0][s] = start_prob[s] * emission_probs[0][s]

    # Viterbi dynamic programming
    for t in range(1, num_timepoints):
        for curr in range(num_states):
            max_prob = 0
            best_prev = 0
            for prev in range(num_states):
                prob = viterbi[t - 1][prev] * transition_matrix[prev][curr] * emission_probs[t][curr]
                if prob > max_prob:
                    max_prob = prob
                    best_prev = prev
            viterbi[t][curr] = max_prob
            backtrack[t][curr] = best_prev

    # Backtrack from the most likely final state (you can force S3 here if needed)
    final_state = 2  # S3
    path_indices = [final_state]
    for t in reversed(range(1, num_timepoints)):
        path_indices.insert(0, backtrack[t][path_indices[0]])

    most_likely_path = [states[i] for i in path_indices]
    final_prob = viterbi[num_timepoints - 1][final_state]

    print("Most likely path to S3 at time 10:", most_likely_path)
    print("Probability of that path:", final_prob)

task_4_viterbi_final()



# task_3()
