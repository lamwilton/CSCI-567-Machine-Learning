import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples

		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################

		features = np.asarray(features)
		if np.ndim(features) == 1:
			features = np.expand_dims(features, axis=0)
		N = features.shape[0]
		self.betas = np.asarray(self.betas)
		pred = np.zeros(N)
		boo = np.array([self.betas[t] * np.asarray(self.clfs_picked[t].predict(features)) for t in range(self.T)]).sum(axis=0)
		pred = np.sign(boo)
		return pred.tolist()
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		features = np.asarray(features)		#initization
		if np.ndim(features) == 1:
			features = np.expand_dims(features, axis=0)
		labels = np.asarray(labels)
		h = list(self.clfs)
		h_list = []
		for n in range(len(h)):
			h_list.append(DecisionStump(h[n].s, h[n].b, h[n].d))		# import classifiers
		N = features.shape[0]
		D = np.zeros(N)		#means D_t, D_t+1
		h_t = []
		e_t = b_t = np.zeros(self.T)
		D[:] = 1 / N
		for t in range(0, self.T):
			h_t1 = np.array([np.multiply(D, (h_list[n].predict(features) != labels).astype(int)).sum() for n in range(len(h))])
			h_t.append(h_list[np.argmin(h_t1)])
			e_t[t] = np.multiply(D, (h_t[t].predict(features) != labels).astype(int)).sum()
			b_t[t] = 0.5 * np.log((1 - e_t[t]) / e_t[t])

			sgn = (h_t[t].predict(features) != labels).astype(int)
			sgn[sgn == 0] = -1
			D = D * np.exp(b_t[t] * sgn)

			D = D / np.sum(D)
		self.clfs_picked = h_t
		self.betas = b_t.tolist()
		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	