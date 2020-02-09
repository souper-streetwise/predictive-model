from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import t
from typing import Union, Sequence
import numpy as np

IntVector = Sequence[int]
FloatVector = Sequence[float]
IntMatrix = Sequence[Sequence[int]]
FloatMatrix = Sequence[Sequence[float]]

class pExtraTreesRegressor(ExtraTreesRegressor):
    ''' ExtraTreesRegressor with confidence intervals. '''

    def __init__(self,
        n_estimators: int = 1000,
        criterion: str = 'mse',
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
        inbag: Union[IntMatrix, None] = None): 

        super(pExtraTreesRegressor, self).__init__(
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

    def fit(self, X, *args, **kwargs):
        preds = super(pExtraTreesRegressor, self).fit(X, *args, **kwargs)
        self.ntrain = X.shape[0]
        self.inbag = self.calculate_inbag()
        return preds

    def get_bootstrap_sample_size(self):
        ''' Compute the size of the bootstrap samples. '''
        if self.max_samples is None: 
            return self.ntrain
        elif isinstance(self.max_samples, int):
            return self.max_samples
        elif isinstance(self.max_samples, float): 
            return int(round(self.ntrain * self.max_samples))

    def get_random_states(self):
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

    def calculate_inbag(self):
        ''' Compute the samples used to create the bagged estimators. '''
        if self.inbag is None:
            self.inbag = np.zeros((self.ntrain, self.n_estimators), 
                dtype = np.int32)

        bootstrap_sample_size = self.get_bootstrap_sample_size()

        def sampler(random_state: np.random.RandomState):
            return random_state.randint(0, self.ntrain, bootstrap_sample_size)

        sample_idxs = [sampler(rnd) for rnd in self.get_random_states()]

        for idx in range(self.n_estimators):
            self.inbag[:, idx] = np.bincount(sample_idxs[idx], 
                minlength = self.ntrain)

        return self.inbag

    def predict(self, X, return_cis: bool = False, alpha: float = 0.99):

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

        if not return_cis:
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
            bootstrap_sample_size = self.get_bootstrap_sample_size()
            bias = bootstrap_sample_size * bootstrap_var.view() 
            bias /= self.n_estimators
            V_IJ_unbiased = V_IJ - bias

            # Compute the standard errors of the predictions
            std_errs = np.sqrt(V_IJ_unbiased / self.n_estimators)

            # Compute the radii of the alpha confidence intervals,
            # using a t-distribution
            t_factor = t.ppf((1 + alpha) / 2., self.ntrain - 1)
            radii = t_factor * std_errs.view()

            # Build the confidence intervals
            intervals = np.empty((ntest, 2), dtype = np.float32)
            intervals[:, 0] = mean_preds - radii
            intervals[:, 1] = mean_preds + radii

            return mean_preds, intervals

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
 
if __name__ == '__main__':
    pass
