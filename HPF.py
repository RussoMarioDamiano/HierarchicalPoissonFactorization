import numpy as np
from scipy.special import digamma
import sys
from tqdm import trange
from sklearn.metrics import mean_squared_error

def mse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return mean_squared_error(prediction, ground_truth)


class HPF():
    """
    Initialize a Hierarchical Poisson Factorization Recommender System.
    Model by Gopalan et al. (2013)

    Parameters:
    ----------
        - K : int
          dimensionality of the latent preferences and qualities vectors.
        - a_1, b_1 : floats
          prior hyperparameters on the Gamma(a_1, a_1/b_1) prior for
          user activity value.
        - a : float
          shape hyperparameter for the Gamma(a, user_activity) prior
          for the elements of user u's preference vector.
        - c_1, d_1 : floats
          prior hyperparameters on the Gamma(c_1, c_1/d_1) prior for
          item popularity value.
        - c : float
          shape hyperparameter for the Gamma(a, item_popularity) prior
          for the elements of item i's qualities vector.
    """

    def __init__(self, K, a_1, b_1, a, c_1, d_1, c):
        self.K = K
        self.a_1 = a_1
        self.b_1 = b_1
        self.a = a
        self.c_1 = c_1
        self.d_1 = d_1
        self.c = c


    def fit(self, epochs, train, val=None):
        """
        Fit a Hierarchical Poisson Factorization Model to
        training data.

        Parameters:
        ----------
            - epochs : int
              number of training epochs.
            - train : numpy.array
              (U X I) array where each row is a user, each column is
              an item.
        """
        # initialize error lists
        self.train_error = []
        self.train = train
        self.val = val
        self.U, self.I = self.train.shape

        if self.val.any():
            self.val_error = []

        # intialize variational parameters to the prior
        self.__initialize_variational_params()

        self.resume_training(epochs)


    def resume_training(self, epochs):
        """
        Resume HPF training for additional epochs.

        Parameters:
        ----------
            - epochs : int
              number of additional training epochs.
        """
        pbar = trange(epochs, file=sys.stdout, desc = "HPF")
        for iteration in pbar:
            # for each each u,i for which the rating is > 0:
            for u, i in zip(self.train.nonzero()[0], self.train.nonzero()[1]):

                # update the variational multinomial parameter
                self.phi[u,i] = [np.exp(digamma(self.gamma_shp[u, k]) - np.log(self.gamma_rte[u, k])
                        + digamma(self.lambda_shp[i, k]) - np.log(self.lambda_rte[i, k])) for k in range(self.K)]
                # normalize the multinomial probability vector
                self.phi[u,i] = self.phi[u,i]/np.sum(self.phi[u,i])

            #for each user, update the user weight and activity parameters
            for u in range(self.U):
                self.gamma_shp[u] = [(self.a + np.sum(self.train[u, :] * self.phi[u,:,k])) for k in range(self.K)]
                self.gamma_rte[u] = [(self.kappa_shp/self.kappa_rte[u] + np.sum(self.lambda_shp[:, k]/self.lambda_rte[:,k])) for k in range(self.K)]
                self.kappa_rte[u] = (self.a_1/self.b_1) + np.sum(self.gamma_shp[u, :]/self.gamma_rte[u, :])

            #for each item, update the item weight and popularity parameters
            for i in range(self.I):
                self.lambda_shp[i] = [(self.c + np.sum(self.train[:, i] * self.phi[:,i,k])) for k in range(self.K)]
                self.lambda_rte[i] = [(self.tau_shp/self.tau_rte[i] + np.sum(self.gamma_shp[:, k]/self.gamma_rte[:,k])) for k in range(self.K)]
                self.tau_rte[i] = (self.c_1/self.d_1) + np.sum(self.lambda_shp[i, :]/self.lambda_rte[i, :])

            # obtain the latent vectors:
            self.theta = self.gamma_shp/self.gamma_rte
            self.beta = self.lambda_shp/self.lambda_rte
            self.prediction = np.dot(self.theta, self.beta.T)

            self.train_error.append(mse(self.prediction, self.train))
            if self.val.any():
                # Note to self: Very misleading measures! Explicit ratings in train here are zero. Useful only for convergence diagnostic purposes!!
                self.val_error.append(mse(self.prediction, self.val))
                pbar.set_description(f"HPF Val MSE: {np.round(self.val_error[-1], 4)} - Progress")
            else:
                pbar.set_description(f"HPF Train MSE: {np.round(self.train_error[-1], 4)} - Progress")


    
    def __initialize_variational_params(self):
        # phi: (U X I X K) matrix of variational parameters for the multinomial
        self.phi = np.zeros(shape=[self.U, self.I, self.K])

        #variational parameter random initialization
        # k_rte: (U X 1) array
        self.kappa_rte = (np.random.rand(self.U) + 1) * self.a_1
        # tau_rte: (I X 1) array
        self.tau_rte = (np.random.rand(self.I) + 1) * self.c_1

        # gamma_shp, gamma_rte: (U X K) numpy arrays
        self.gamma_shp = np.random.gamma(shape=self.a_1, scale=(self.b_1/self.a_1), size=(self.U, self.K))
        self.gamma_rte = (np.random.rand(self.U, self.K) + 1) * self.a
        # lambda_shp, lambda_rte: (I X K) numpy arrays
        self.lambda_shp = np.random.gamma(shape=self.c_1, scale=(self.c_1/self.d_1), size=(self.I, self.K))
        self.lambda_rte = (np.random.rand(self.I, self.K) + 1) * self.c

        # k_shp, tau_shp intialization rules come from the CAVI algorithm and do not need to be updated at each iteration.
        # these values are constant for each u, i respectively, so they are a scalar.
        self.kappa_shp = self.a_1 + (self.K * self.a)
        self.tau_shp = self.c_1 + (self.K * self.c)
