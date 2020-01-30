from sklearn.ensemble import ExtraTreesRegressor

class pExtraTreesRegressor(ExtraTreesRegressor):
    def predict(self, X, percentiles: list = None):
        from sklearn.utils.validation import check_is_fitted
        import numpy as np

        check_is_fitted(self)

        # Check data
        X = self._validate_X_predict(X)

        if self.n_outputs_ > 1:
            y_hats = np.zeros((X.shape[0], self.n_outputs_, 
                len(self.estimators_)), dtype = np.float64)
        else:
            y_hats = np.zeros((X.shape[0], len(self.estimators_)), 
                dtype = np.float64)

        for idx, estimator in enumerate(self.estimators_):
            if self.n_outputs_ > 1:
                y_hats[:, :, idx] = estimator.predict(X, check_input = False)
            else:
                y_hats[:, idx] = estimator.predict(X, check_input = False)

        if percentiles is None:
            return np.mean(y_hats, axis = -1)
        else:
            return np.stack([np.percentile(y_hats, p, axis = -1)
                for p in percentiles], axis = -1)

if __name__ == '__main__':
    pass
