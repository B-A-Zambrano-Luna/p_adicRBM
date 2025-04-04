"""P-adic based Restricted Boltzmann Machine
"""

# from msilib.schema import Component
# from re import A
import time
from sklearn.base import BaseEstimator
import numpy as np
import scipy.sparse as sp
from scipy.special import expit  # logistic function

from sklearn.utils.extmath import safe_sparse_dot

from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import log_logistic
from numpy import linalg as LA
import os
import math


class Padic_BernoulliRBM(BaseEstimator):
    """Bernoulli Restricted Boltzmann Machine (RBM).

    A p-adic based Restricted Boltzmann Machine with binary visible units and
    binary hidden units. Parameters are estimated using Stochastic Maximum
    Likelihood (SML), also known as Persistent Contrastive Divergence (PCD).

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components. TODO: change the time complexity

    Read more in the :ref:`User Guide <rbm>`.

    Parameters
    ----------
    n_components : int, default=256
        Number of binary hidden units.

    learning_rate : float, default=0.1
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, default=10
        Number of examples per minibatch.

    n_iter : int, default=10
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : int, default=0
        The verbosity level. The default, zero, means silent mode. Range
        of values is [0, inf].

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for:

        - Gibbs sampling from visible and hidden layers.

        - Initializing components, sampling from layers during fit.

        - Corrupting the data when scoring samples.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    intercept_hidden_ : array-like of shape (n_components,)
        Biases of the hidden units.

    intercept_visible_ : array-like of shape (n_features,)
        Biases of the visible units.

    components_ : array-like of shape (n_components, n_features)
        Weight matrix, where `n_features` is the number of
        visible units and `n_components` is the number of hidden units.

    h_samples_ : array-like of shape (batch_size, n_components)
        Hidden Activation sampled from the model distribution,
        where `batch_size` is the number of examples per minibatch and
        `n_components` is the number of hidden units.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
    sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
    sklearn.decomposition.PCA : An unsupervised linear dimensionality
        reduction model.

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import BernoulliRBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = BernoulliRBM(n_components=2)
    >>> model.fit(X)
    BernoulliRBM(n_components=2)
    """

    def __init__(
        self,
        n_components=4096,  # number of hidden units``
        h_components=4096,
        *,
        learning_rate=0.1,
        batch_size=16,
        n_iter=10,
        verbose=0,
        random_state=None,
        rad_supp=0,
        args,
    ):
        self.n_components = n_components
        self.h_components = h_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.updates = []
        self.is_best = False
        self.args = args
        self.energy = []
        self.rad_supp = rad_supp
        self.G_l = int(math.log(n_components, args.p))
        self.G_s = int(math.log(h_components, args.p))

    def transform(self, X):
        """Compute the hidden layer activation probabilities, P(h=1|v=X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Latent representations of the data.
        """
    #     check_is_fitted(self)

    #     X = self._validate_data(
    #         X, accept_sparse="csr", reset=False, dtype=(np.float64, np.float32)
    #     )
        return self._mean_hiddens(X)

    def _mean_hiddens(self, v):
        """Computes the probabilities {P(h_i=1|v)}_{i=1}^N.

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        # p = np.zeros(v.shape, dtype=v.dtype)
        # for j in range(v.shape[1]):
        #     a = np.dot(v, np.roll(self.components_,-j)) + self.intercept_hidden_[j] - 0.5
        #     p[:,j] = a.squeeze()
        h_comp = self.h_components
        n_comp = self.n_components
        p = [np.dot(v, self.roll(self.components_, j % n_comp)) +
             self.intercept_hidden_[j] for j in range(0, h_comp)]
        p = np.transpose(np.array(p))
        # p.shape #(50, 729)
        return expit(p)

    def _sample_hiddens(self, v, rng):
        """Sample from the distribution P(h|v).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to sample from.

        rng : RandomState instance
            Random number generator to use.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer.
        """
        p = self._mean_hiddens(v)
        return rng.uniform(size=p.shape) < p

    def _sample_visibles(self, h, rng):
        """Sample from the distribution P(v|h).

        Parameters
        ----------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        rng : RandomState instance
            Random number generator to use.

        Returns
        -------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.
        """

        # p = np.zeros(h.shape, dtype=h.dtype)
        # for j in range(h.shape[1]):
        #     a = safe_sparse_dot(h, np.roll(np.flip(self.components_),j)) + self.intercept_visible_[j] - 0.5
        #     p[:,j] = a.squeeze()
        h_comp = self.h_components
        n_comp = self.n_components

        p = []
        for t in range(n_comp):
            w_t = self.roll(np.flip(self.components_), t)
            ext_w_t = np.zeros(h.shape[1])
            for l in range(int(h_comp/n_comp)):
                ext_w_t[n_comp*l: n_comp*(l+1)] = w_t
            p.append(safe_sparse_dot(h, ext_w_t) +
                     self.intercept_visible_[t])
        # p = [safe_sparse_dot(h, self.roll(np.flip(self.components_), j)) +
        #       self.intercept_visible_[j] for j in range(h.shape[1])]

        p = np.transpose(np.array(p))
        p = expit(p)
        return rng.uniform(size=p.shape) < p

    def _free_energy(self, v):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : ndarray of shape (n_samples,)
            The value of the free energy.
        """

        # s: array like (n_samples,) 1D
        s = -safe_sparse_dot(v, self.intercept_visible_) + \
            0.5*LA.norm(v, axis=1)**2
        a = [safe_sparse_dot(v, self.roll(self.components_, j % v.shape[1])) +
             self.intercept_hidden_[j] for j in range(self.h_components)]
        a = np.array(a)
        s -= np.sum(np.logaddexp(0, a), axis=0)
        # for j in range(v.shape[1]):
        #     a = safe_sparse_dot(v, self.roll(self.components_,j))+self.intercept_hidden_[j]-0.5 ##s: array like (n_samples, 1) 2D
        #     s -= np.logaddexp(0,a).squeeze()
        self.energy.append(s)
        return s
        # x2 =
        # array([[0, 1, 2, 3, 4],
        #     [5, 6, 7, 8, 9]])

        # self.roll(x2, 1, axis=1)
        # array([[4, 0, 1, 2, 3],
        #     [9, 5, 6, 7, 8]])

        # np.roll(x2, -1, axis=1)
        # array([[1, 2, 3, 4, 0],
        #     [6, 7, 8, 9, 5]])

        # c = np.array([[ 1, 2, 3],
        # ...               [-1, 1, 4]])
        # >>> LA.norm(c, axis=0)
        # array([ 1.41421356,  2.23606798,  5.        ])
        # >>> LA.norm(c, axis=1)
        # array([ 3.74165739,  4.24264069])
        # >>> LA.norm(c, ord=1, axis=1)
        # array([ 6.,  6.])
        #
        # self.intercept_visible_)
        # - np.logaddexp(0, safe_sparse_dot(v, self.components_.T) + self.intercept_hidden_).sum(axis=1)
        # np.logaddexp(x1,x2) = log(exp(x1) + exp(x2))

    def gibbs(self, v):
        """Perform one Gibbs sampling step.
        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to start from.

        Returns
        -------
        v_new : ndarray of shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
        # check_is_fitted(self)
        h_ = self._sample_hiddens(v, self.random_state_)
        v_ = self._sample_visibles(h_, self.random_state_)
        return v_

    def gibbs_sampling(self, k, x):
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        counter = 0
        while counter < k:
            # return T/F which original pixels are activated
            gibbs_x = self.gibbs(x)
            x = np.zeros_like(x)
            x[gibbs_x] = 1  # make the "turned on" pixels to be 1, others to be 0
            counter += 1
        return x

    # def partial_fit(self, X, y=None):
    #     """Fit the model to the partial segment of the data X.

    #     Parameters
    #     ----------
    #     X : ndarray of shape (n_samples, n_features)
    #         Training data.

    #     y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
    #         Target values (None for unsupervised transformations).

    #     Returns
    #     -------
    #     self : BernoulliRBM
    #         The fitted model.
    #     """
    #     first_pass = not hasattr(self, "components_")
    #     X = self._validate_data(
    #         X, accept_sparse="csr", dtype=np.float64, reset=first_pass
    #     )
    #     if not hasattr(self, "random_state_"):
    #         self.random_state_ = check_random_state(self.random_state)
    #     if not hasattr(self, "components_"):
    #         self.components_ = np.asarray(
    #             self.random_state_.normal(0, 0.01, (self.n_components, X.shape[1])),
    #             order="F",
    #         )
    #         self._n_features_out = self.components_.shape[0]
    #     if not hasattr(self, "intercept_hidden_"):
    #         self.intercept_hidden_ = np.zeros(
    #             self.n_components,
    #         )
    #     if not hasattr(self, "intercept_visible_"):
    #         self.intercept_visible_ = np.zeros(
    #             X.shape[1],
    #         )
    #     if not hasattr(self, "h_samples_"):
    #         self.h_samples_ = np.zeros((self.batch_size, self.n_components))

    #     self._fit(X, self.random_state_)

    def _fit(self, v_pos, rng):
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        v_pos : ndarray of shape (n_samples=batch_size, n_features)
            The data to use for training.

        rng : RandomState instance
            Random number generator to use for sampling.
        """
        start = time.time()

        h_pos = self._mean_hiddens(v_pos)
        v_neg = v_pos
        for i in range(self.args.k):
            self.h_samples_ = self._sample_hiddens(v_neg, rng)
            v_neg = self._sample_visibles(self.h_samples_, rng)

        h_neg = self._mean_hiddens(v_neg)
        self.h_samples_ = self._sample_hiddens(v_neg, rng)

        lr = float(self.learning_rate) / v_pos.shape[0]
        # print('time spend on update{}'.format(time.time()-start))

        # print('lr=',lr)
        # update =  np.zeros((self.n_components,1))

        # for k in range(self.n_components):
        #     update[k] = np.vdot(v_pos, np.roll(h_pos,-k,axis=1))
        #     update[k] -= np.vdot(v_neg, np.roll(h_neg,-k,axis=1))
        # start = time.time()
        p = self.args.p

        h_comp = self.h_components
        n_comp = self.n_components

        update = []
        for k in range(p**(self.rad_supp)):
            v_pos_k = self.roll(v_pos, -k, axis=1)
            v_neg_k = self.roll(v_neg, -k, axis=1)
            ext_v_pos_k = np.zeros(h_pos.shape)
            ext_v_neg_k = np.zeros(h_neg.shape)
            for l in range(int(h_comp/n_comp)):
                ext_v_pos_k[:, n_comp*l: n_comp*(l+1)] = v_pos_k
                ext_v_neg_k[:, n_comp*l: n_comp*(l+1)] = v_neg_k

            update.append(np.vdot(h_pos, ext_v_pos_k)
                          - np.vdot(h_neg, ext_v_neg_k))
        # print('time spend on update{}'.format(time.time()-start))
        update = np.array(update)
        update.resize(self.n_components)
        # print(update[:10], "update")
        # import matplotlib.pyplot as plt
        # A = update.copy()
        # A.resize((27, 27))
        # plt.imshow(A)
        # plt.title("Update")
        # plt.show()

        # print('update[k]',update[k])
        self.updates.append(LA.norm(update)/np.sqrt(v_pos.shape[0]))
        self.components_ += lr * update
        # print('update shape',update.squeeze().shape, self.components_.shape)

        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (
            np.asarray(v_pos.sum(axis=0)).squeeze() - v_neg.sum(axis=0)
        )

        # h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # update negative state
        # h_neg = rng.uniform(size=h_neg.shape) < h_neg
        # self.h_samples_ = np.floor(h_neg, h_neg) #the second h_neg is used to store the results TODO

    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : ndarray of shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        check_is_fitted(self)

        v = self._validate_data(X, accept_sparse="csr", reset=False)
        rng = check_random_state(self.random_state)

        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]), rng.randint(0, v.shape[1], v.shape[0]))
        #random.RandomState.randint(low, high=None, size=None, dtype=int)
        # np.arange(3) # array([0, 1, 2])
        if sp.issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]

        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return v.shape[1] * log_logistic(fe_ - fe)  # what is this?
        # log_logistic Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.

    def fit(self, X, y=None):
        """Fit the model to the data X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. list type

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = self._validate_data(X, accept_sparse="csr",
                                dtype=(np.float64, np.float32))
        # """Validate input data and set or check the `n_features_in_` attribute.
        n_samples = X.shape[0]
        if X.shape[1] != pow(self.args.p, 2*self.args.l):
            print('the visible states has wrong size. quit.')
            quit()
        rng = check_random_state(self.random_state)
        """
        def check_random_state(seed):
        Turn seed into a np.random.RandomState instance.
        Parameters
        ----------
        seed : None, int or instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.
        Returns
        -------
        """
        p = self.args.p
        self.components_ = np.asarray(
            rng.normal(0, 2.0, (p**(self.rad_supp), )),
            order="F",
            dtype=X.dtype,
        )
        self.components_.resize(self.n_components)
        # components_ is used to store the weights for w_k
        #random.RandomState.normal(mean=0.0, sd=1.0, output_size=None)
        # np.asarray: Convert the input to an array.  ‘F’ column-major (Fortran-style) memory representation.
        # _n_features_out = # of hidden units
        self._n_features_out = self.components_.shape[0]
        # self.intercept_hidden_ = np.zeros(self.n_components, dtype=X.dtype)  #generate a zero array
        # self.intercept_visible_ = np.zeros(X.shape[1], dtype=X.dtype)

        self.intercept_hidden_ = np.asarray(
            rng.normal(0, 2.0, (self.h_components, )),
            order="F",
            dtype=X.dtype,
        )
        self.intercept_visible_ = np.asarray(
            rng.normal(0, 2.0, (self.n_components, )),
            order="F",
            dtype=X.dtype,
        )

        # why set to zero at first?
        self.h_samples_ = np.zeros(
            (self.batch_size, self.h_components), dtype=X.dtype)

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(
            gen_even_slices(n_batches * self.batch_size,
                            n_batches, n_samples=n_samples)
        )

        """
        def gen_even_slices(n, n_packs, *, n_samples=None):
        Generator to create n_packs slices going up to n.
        Parameters
        ----------
        n : int
        n_packs : int
            Number of slices to generate.
        n_samples : int, default=None
            Number of samples. Pass n_samples when the slices are to be used for
            sparse matrix indexing; slicing off-the-end raises an exception, while
            it works for NumPy arrays.
        Yields
        ------
        slice
        See Also
        --------
        gen_batches: Generator to create slices containing batch_size elements
            from 0 to n.
        Examples
        --------
        >>> from sklearn.utils import gen_even_slices
        >>> list(gen_even_slices(10, 1))
        [slice(0, 10, None)]
        >>> list(gen_even_slices(10, 10))
        [slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
        >>> list(gen_even_slices(10, 5))
        [slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
        >>> list(gen_even_slices(10, 3))
        [slice(0, 4, None), slice(4, 7, None), slice(7, 10, None)]

        a = ("a", "b", "c", "d", "e", "f", "g", "h")
        x = slice(3, 5)
        print(a[x])
        >>> ('d', 'e')
        """
        verbose = self.verbose
        # import matplotlib.pyplot as plt
        # A = self.components_.copy()
        # A.resize((27, 27))
        # plt.imshow(A)
        # plt.title("Initial")
        # plt.show()
        for iteration in range(1, self.n_iter + 1):
            begin = time.time()
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], rng)
                A = self.components_.copy()
                A.resize((27, 27))
                # plt.imshow(A)
                # plt.title("iteration =")
                # plt.show()
            end = time.time()
            print("at iteration:%d, time:%.2fs " % (iteration, end - begin))

            if len(self.updates) > 50:
                a = self.updates[-50:]
            else:
                a = np.array(self.updates)
            norm_a = LA.norm(a)

            self.save(iteration, is_best=self.is_best)

            # if norm_a < 1e-7:
            #     self.is_best = True
            #     self.terminate(norm_a)
            #     break

            if verbose:
                print(
                    "[%s] Iteration %d, pseudo-likelihood = %.2f, time = %.2fs"
                    % (
                        type(self).__name__,
                        iteration,
                        self.score_samples(X).mean(),
                        end - begin,
                    )
                )
                # begin = end

            # self.terminate()
        self.save_final()
        return self

    def sample_completion(self, v, m):
        rng = check_random_state(self.random_state)
        n_samples = len(v)
        v0 = np.asarray(v)
        for i in range(m):
            h0 = self._sample_hiddens(v0, rng)
            v0 = self._sample_visibles(h0, rng)
        return v0

    # TODO add the terminate function based on updates
    def save_final(self):
        np.savetxt('./results/updates.txt', self.updates, delimiter=',')
        np.savetxt('./results/free_energy.txt', self.energy, delimiter=',')

    def save(self, epoch, is_best=False):
        apath = './saved_models/'
        save_dirs = [apath]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best'))
        for s in save_dirs:  # torch.save(model.state_dict(), PATH)
            np.savetxt(os.path.join(s, 'model_epoch{}_w.csv'.format(
                epoch)), self.components_, delimiter=',')
            np.savetxt(os.path.join(s, 'model_epoch{}_a_visible.csv'.format(
                epoch)), self.intercept_visible_, delimiter=',')
            np.savetxt(os.path.join(s, 'model_epoch{}_b_hidden.csv'.format(
                epoch)), self.intercept_hidden_, delimiter=',')

            #print('model saving success')

    def roll(self, V, k, axis=1):
        # this roll function  fullfiles the following functionality:
        #V_new[i+k] = v[i]
        if V.ndim == 1:
            N = V.shape[0]
            if k < 0 and k > -N:
                k = N + k
            return np.concatenate((V[N-k:], V[:N-k]))
        elif V.ndim == 2:
            if axis == 0:
                V = np.array(V).transpose()

            N = V.shape[1]
            if k < 0 and k > -N:
                k = N + k
            if axis == 0:
                return np.concatenate((V[:, N-k:], V[:, :N-k]), axis=1).transpose()
            else:
                return np.concatenate((V[:, N-k:], V[:, :N-k]), axis=1)
        else:
            print('this code does not handle hdim>=3 array yet.')
            quit()
