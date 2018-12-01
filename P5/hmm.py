from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  alpha[:, 0] = np.multiply(pi, B[:, O[0]])
  for t in range(1, N):
    alpha[:, t] = B[:, O[t]] * np.multiply(A.T, alpha[:, t - 1]).sum(axis=1)
  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  beta[:, N-1] = 1
  for t in range(N-2, -1, -1):
    bb = np.multiply(B[:, O[t+1]], beta[:, t+1])
    beta[:, t] = np.multiply(A, bb).sum(axis=1)
  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward messages

  Inputs:
  - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})

  Returns:
  - prob: A float number of P(X_{1:N}=O)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  t = alpha.shape[1] - 1
  prob = alpha[:, t].sum()
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward messages

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(X_{1:N}=O)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  alpha1 = np.multiply(pi, B[:, O[0]])
  prob = np.multiply(beta[:, 0], alpha1).sum()
  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path (in terms of the state index)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  ###################################################
  '''
  pi = np.array([0.6,0.4])
  B = np.array([[0.5,0.4], [0.1, 0.3]])
  A= np.array([[0.7,0.3], [0.4,0.6]])
  O = np.array([0,1])
  '''
  S = len(pi)
  N = len(O)
  delta = np.zeros([S, N])
  Delta = np.zeros([S, N])
  path = np.zeros(N)
  delta[:, 0] = np.multiply(pi, B[:, O[0]])
  for t in range(1, N):
    adelta = np.multiply(A.T, delta[:, t - 1])
    maxs = np.amax(adelta, axis=1)
    delta[:, t] = np.multiply(B[:, O[t]], maxs)
    Delta[:, t] = np.argmax(adelta, axis=1)
  path[N-1] = np.argmax(delta[:, N-1])
  for t in range(N-1, 0, -1):
    path[t-1] = Delta[path[t].astype(int), t]
  path = path.astype(int)
  path = path.tolist()
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()