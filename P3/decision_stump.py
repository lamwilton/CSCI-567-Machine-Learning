import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
	def __init__(self, s:int, b:float, d:int):
		self.clf_name = "Decision_stump"
		self.s = s
		self.b = b
		self.d = d

	def train(self, features: List[List[float]], labels: List[int]):
		pass
		
	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		##################################################
		# TODO: implement "predict"
		##################################################
		features = np.asarray(features)
		if np.ndim(features) == 1:
			features = np.expand_dims(features, axis=0)
		N = features.shape[0]
		pred = np.zeros(N)
		for n in range(0, N):
			if features[n, self.d] > self.b:
				pred[n] = self.s
			else:
				pred[n] = - self.s
		return pred.tolist()