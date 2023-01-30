from functools import cached_property
import numpy as np
from scipy import optimize
from scipy.stats import gamma


class ModifiedKernel:
    def __init__(self, kernel: "Kernel"):
        self.kernel = kernel

    def _modify_weights(self, feature_weights):
        return 1/(1+1/feature_weights)

    def feature_map(self, data):
        """Delegated to the original kernel.
        Params:
            data: (n_sample,)
        Returns:
            (n_feat, n_sample)
        """
        return self.kernel.feature_map(data)

    @cached_property
    def modified_weights(self):
        return self._modify_weights(self.kernel.feature_weights)

    def gram_matrix(self, features):
        """
        Params:
            features: (n_feat, n_sample)
        Returns:
            (n_sample, n_sample)
        """
        modified_feature_weights = self.modified_weights
        return features.T @ np.diag(modified_feature_weights) @ features


class AlphaObjective:
    def __init__(self, gram_matrix):
        self.tildeK = gram_matrix
        # self.eps = 1e-5
        self.eps=0
        self.n = gram_matrix.shape[0]

    def objective(self, alpha) -> float:
        """
        Params:
            alpha : 1-d array with with shape (n,)
        Refs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        """
        return np.sum(np.log((alpha + self.eps) ** 2)) + 1 / 2 * alpha @ self.tildeK @ alpha.T

    def grad(self, alpha):
        """
        Params:
            alpha : 1-d array with with shape (n,)
        Refs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        """
        return 2 / (alpha+self.eps) + alpha @ self.tildeK

    def init_params(self):
        return np.ones(self.n) / self.n


class Method:
    def __init__(self, mod_kernel: ModifiedKernel):
        self.mod_kernel = mod_kernel

    def find_alpha(self, features):
        tildeK = self.mod_kernel.gram_matrix(features)
        obj = AlphaObjective(tildeK)
        result = optimize.minimize(obj.objective, obj.init_params(), jac=obj.grad, method='BFGS', tol=1e-2)
        print(result)
        return result.x

    def laplace_mean(self, eval_data, features, alpha):
        """ Mean of the predictive distribution of the posterior GP evaluated at ``eval_data``.
        Params:
            eval_data: (n_eval_sample,)
            features: (n_feat, n_sample)
            alpha: (n_sample,)
        Returns:
            (n_eval_sample,)
        """
        eval_features = self.mod_kernel.feature_map(eval_data)
        return eval_features.T @ np.diag(self.mod_kernel.modified_weights) @ features @ alpha

    def laplace_variance(self, eval_data, features, alpha):
        """ Variance of the predictive distribution of the posterior GP evaluated at ``eval_data``.
        Params:
            eval_data: (n_eval_sample,)
            features: (n_feat, n_sample)
            alpha: (n_sample,)
        Returns:
            (n_eval_sample,)
        """
        # (n_feat, n_eval_sample)
        eval_features = self.mod_kernel.feature_map(eval_data)

        # ~k(x*, X)
        k_eval_train = eval_features.T @ np.diag(self.mod_kernel.modified_weights) @ features
        # S = (~k(X,X) . (alpha alpha^T) + 2I)
        S = self.mod_kernel.gram_matrix(features) * (alpha @ alpha.T) + 2 * np.eye(features.shape[1])

        # ~k(x*, x*)
        # eval_features: (n_feat, n_eval_sample)
        # modified_weights: (n_feat,)
        print(eval_features.shape)
        print(self.mod_kernel.modified_weights.shape)
        k_eval_eval = (eval_features ** 2 * self.mod_kernel.modified_weights[:,None]).sum(axis=0) # (n_eval_sample,)
        # ~k(x*,x*) - (~k(x*,X) . alpha) S^{-1} (alpha^T . ~k(X,x*))
        _ka = k_eval_train * alpha # (n_eval_sample,n_sample)
        sigma_sq = k_eval_eval - np.diag(np.einsum('xi,ij,jy->xy', _ka, (np.linalg.inv(S)), _ka.T))
        return sigma_sq

    def marginal_likelihood(self, features, alpha):
        _gram_matrix = self.mod_kernel.gram_matrix(features)
        print(alpha ** 2)
        log_h = - np.sum(np.log(2 * (alpha ** 2)))
        quadratic_term = alpha @ _gram_matrix @ alpha.T
        Nu = np.sum(np.log(1 / (1 + self.mod_kernel.kernel.feature_weights)))
        logdet = np.log(np.linalg.det(_gram_matrix * (alpha.T @ alpha) + 2 * np.eye(features.shape[1])))
        c = features.shape[1] * np.log(2)
        print('log_h', log_h, 'quad', quadratic_term, 'Nu', Nu, 'logdet', logdet, 'c', c)
        return log_h - .5 * (quadratic_term + Nu - logdet + c)

    def predict(self, eval_data, features, alpha):
        mu = self.laplace_mean(eval_data, features, alpha)
        sigma_sq = self.laplace_variance(eval_data, features, alpha)
        alpha = (mu**2 + sigma_sq) ** 2 / (2 * sigma_sq * (2 * mu ** 2 + sigma_sq))
        beta = (2 * mu ** 2 * sigma_sq + sigma_sq ** 2) / (mu ** 2 + sigma_sq)
        result = []
        for _alpha, _beta in zip(alpha, beta):
            result.append(gamma.ppf([.1, .5, .9], _alpha, scale=_beta))
        return result
