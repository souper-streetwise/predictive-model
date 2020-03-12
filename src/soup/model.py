from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import t
from typing import Union, Sequence
import numpy as np
import pandas as pd

IntVector = Sequence[int]
FloatVector = Sequence[float]
IntMatrix = Sequence[Sequence[int]]
FloatMatrix = Sequence[Sequence[float]]

class pExtraTreesRegressor(ExtraTreesRegressor):
    ''' ExtraTreesRegressor with confidence intervals. '''

    def __init__(self,
        n_estimators: int = 10000,
        criterion: str = 'mae',
        max_depth: Union[int, None] = None,
        min_samples_split: Union[int, float, None] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.,
        max_features: Union[int, float, str, None] = 'auto',
        max_leaf_nodes: Union[int, None] = None,
        min_impurity_decrease: float = 0.,
        min_impurity_split: float = None,
        oob_score: bool = False,
        n_jobs: Union[int, None] = -1,
        random_state = None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float, None] = None,
        ntrain: Union[int, None] = None,
        inbag: Union[IntMatrix, None] = None,
        residuals: Union[FloatMatrix, None] = None):

        super().__init__(
            n_estimators = n_estimators,
            criterion = criterion,
            max_depth = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            min_weight_fraction_leaf = min_weight_fraction_leaf,
            max_features = max_features,
            max_leaf_nodes = max_leaf_nodes,
            min_impurity_decrease = min_impurity_decrease,
            min_impurity_split = min_impurity_split,
            bootstrap = True,
            oob_score = oob_score,
            n_jobs = n_jobs,
            random_state = random_state,
            verbose = verbose,
            warm_start = warm_start,
            ccp_alpha = ccp_alpha,
            max_samples = max_samples)

        self.ntrain = ntrain
        self.inbag = inbag
        self.residuals = residuals

    def fit(self, X, y, *args, **kwargs):
        preds = super().fit(X, y, *args, **kwargs)
        self.ntrain = X.shape[0]
        self.inbag = self.__calculate_inbag()
        self.residuals = self.__calculate_residuals(X, y)
        return preds

    def __get_bootstrap_sample_size(self):
        ''' Compute the size of the bootstrap samples. '''
        if self.max_samples is None: 
            return self.ntrain
        elif isinstance(self.max_samples, int):
            return self.max_samples
        elif isinstance(self.max_samples, float): 
            return int(round(self.ntrain * self.max_samples))

    def __get_random_states(self):
        ''' Compute random states for all tree estimators. '''
        rnd_states: Sequence[np.random.RandomState] = []
        for tree in self.estimators_:
            seed = tree.random_state
            if seed is None or seed is np.random:
                rnd_states.append(np.random.mtrand._rand)
            if isinstance(seed, int):
                rnd_states.append(np.random.RandomState(seed))
            if isinstance(seed, np.random.RandomState):
                rnd_states.append(seed)
        return rnd_states

    def __calculate_inbag(self):
        ''' Compute the samples used to create the bagged estimators. '''
        self.inbag = np.zeros((self.ntrain, self.n_estimators), 
            dtype = np.int32)

        bootstrap_sample_size = self.__get_bootstrap_sample_size()

        def sampler(random_state: np.random.RandomState):
            return random_state.randint(0, self.ntrain, bootstrap_sample_size)

        sample_idxs = [sampler(rnd) for rnd in self.__get_random_states()]

        for idx in range(self.n_estimators):
            self.inbag[:, idx] = np.bincount(sample_idxs[idx], 
                minlength = self.ntrain)

        return self.inbag

    def __calculate_residuals(self, X, y):

        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values

        # Estimate testing residuals on out-of-bag predictions
        test_residuals = np.zeros((self.ntrain, self.n_estimators))
        for b, tree in enumerate(self.estimators_):
            idxs = np.arange(self.ntrain)[self.inbag[:, b] == 0]
            test_residuals[idxs, b] = y[idxs] - tree.predict(X[idxs, :])

        # Compute the mean residual per observation
        test_residuals = test_residuals.sum(1) / (test_residuals!=0).sum(1)

        # Compute training residuals
        preds = np.zeros((self.ntrain, self.n_estimators), dtype = np.float32)
        for idx, tree in enumerate(self.estimators_):
            preds[:, idx] = tree.predict(X)
        mean_preds = preds.mean(1)
        train_residuals = y - mean_preds

        # Take percentiles of the training- and validation residuals to enable 
        # comparisons between them
        test_residuals = np.percentile(test_residuals, np.arange(0,100,0.1))
        train_residuals = np.percentile(train_residuals, np.arange(0,100,0.1))

        # Compute the .632+ bootstrap estimate
        no_info_err = np.mean(np.abs(np.random.permutation(y) - \
            np.random.permutation(mean_preds)))
        generalisation = np.abs(test_residuals - train_residuals)
        no_info_val = np.abs(no_info_err - train_residuals)
        relative_overfitting_rate = np.mean(generalisation / no_info_val)
        weight = .632 / (1 - .368 * relative_overfitting_rate)
        self.residuals = (1-weight) * train_residuals + weight * test_residuals

        return self.residuals

    def predict(self, X, return_intervals: bool = False, alpha: float = 0.99):
        ''' Predict demand from weather- and date features. '''
        ntest: int
        preds: FloatMatrix
        mean_preds: FloatVector

        # Initialise the number of test samples and the prediction matrix
        ntest = X.shape[0]
        preds = np.zeros((ntest, self.n_estimators), dtype = np.float32)

        # Compute the prediction for every bagged estimator
        for idx, estimator in enumerate(self.estimators_):
            preds[:, idx] = estimator.predict(X)

        # Compute the predictions
        mean_preds = np.mean(preds, axis = 1)

        if not return_intervals:
            mean_preds = np.round(mean_preds).astype(np.int32)
            if mean_preds.shape[0] == 1:
                return mean_preds[0]
            else:
                return mean_preds

        else:
            cpreds: FloatVector 
            covariances: FloatMatrix
            bootstrap_var: FloatVector
            bias: FloatVector
            V_IJ: FloatVector
            V_IJ_unbiased: FloatVector
            std_errs: FloatVector
            radii: FloatVector
            intervals: FloatMatrix
            bootstrap_sample_size: int
            t_factor: float

            # Center the predictions
            cpreds = preds.view() - mean_preds[:, None]

            # Compute the variance estimate
            covariances = ((self.inbag.view() - 1) @ cpreds.view().T)
            covariances /= self.n_estimators
            V_IJ = np.sum(covariances.view() ** 2, axis = 0)

            # Compute the bias correction
            bootstrap_var = np.mean(cpreds.view() ** 2, axis = 1)
            bootstrap_sample_size = self.__get_bootstrap_sample_size()
            bias = bootstrap_sample_size * bootstrap_var.view() 
            bias /= self.n_estimators
            V_IJ_unbiased = V_IJ - bias

            # Compute the standard errors of the predictions
            std_errs = np.sqrt(V_IJ_unbiased * (1 + 1 / self.n_estimators))

            # Compute the radii of the alpha confidence intervals,
            # using a t-distribution
            t_factor = t.ppf((1 + alpha) / 2., self.ntrain - 1)
            radii = t_factor * std_errs.view()

            # Get residual noise
            lower_noise = np.quantile(self.residuals, (1 - alpha) / 2.)
            upper_noise = np.quantile(self.residuals, (1 + alpha) / 2.)

            # Build the prediction intervals
            intervals = np.empty((ntest, 2), dtype = np.float32)
            intervals[:, 0] = mean_preds - radii + lower_noise
            intervals[:, 1] = mean_preds + radii + upper_noise

            # Enforce lower bound to be positive
            intervals[:, 0] = np.maximum(intervals[:, 0],
                np.zeros(intervals.shape[0]))

            mean_preds = np.round(mean_preds).astype(np.int32)
            intervals = np.round(intervals).astype(np.int32)
            if mean_preds.shape[0] == 1:
                return mean_preds[0], tuple(intervals[0])
            else:
                return mean_preds, intervals

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
 
if __name__ == '__main__':
    pass
