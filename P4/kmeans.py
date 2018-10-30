import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        ''' 
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        K = self.n_cluster
        mu_k = np.zeros([K, D])
        mu_k = x[np.random.choice(N, K, replace=True), :]
        J = 10^10
        for iter in range(self.max_iter):
            r_ik = np.zeros([N, K])
            dists = np.zeros([N, K])
            for k in range(K):
                dists[:, k] = np.square(np.linalg.norm(mu_k[k, :] - x, axis=1))     # dists = ||mu_k - x_i|| ^2
            r_ik[np.arange(N), np.argmin(dists, axis=1)] = 1        # Update r_ik to 1 for closest distance
            J_new = np.multiply(r_ik, dists).sum() / N
            if np.absolute(J - J_new) < self.e:
                number_of_updates = iter
                break    # STOP
            J = J_new
            for k in range(0, K):
                if r_ik[:, k].sum() != 0:   # Do not update if points assigned to a cluster
                    mu_k[k] = np.multiply(r_ik[:, k].T, x.T).T.sum(axis=0) / r_ik[:, k].sum()   # Have to transpose twice for broadcasting rules
        y = np.argmin(dists, axis=1)
        return mu_k, y, number_of_updates
        #raise Exception(
        #    'Implement fit function in KMeans class (filename: kmeans.py)')
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement fit function in KMeansClassifier class (filename: kmeans.py)')
        K = self.n_cluster
        centroid_labels = np.zeros(K)
        centroids, assign, i = KMeans.fit(self, x)
        for k in range(K):
            votes = y[np.nonzero(assign == k)]      # Find what label each cluster votes for
            centroid_labels[k] = np.bincount(votes).argmax()
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement predict function in KMeansClassifier class (filename: kmeans.py)')
        K = self.n_cluster
        dists = np.zeros([N, K])
        assignment = np.zeros(N)
        labels = np.zeros(N)
        for k in range(K):
            dists[:, k] = np.square(np.linalg.norm(x - self.centroids[k], axis=1))  # Compute distances for all centroids
        assignment = np.argmin(dists, axis=1)
        labels = self.centroid_labels[assignment]
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

