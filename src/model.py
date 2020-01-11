def train_model(X, y, n_estimators: int = 500, cv: int = 10,
    return_feature_importances: bool = False):
    ''' Train a random forest. '''
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators = n_estimators)
    scores = -cross_val_score(model, X, y, cv = cv)
    output = {'model': model, 'scores': scores}

    if return_feature_importances:
        model.fit(X, y)
        feat_dict = dict(list(zip(model.feature_importances_, X.columns)))
        feat_sorted = sorted(feat_dict, reverse = True)
        feat_imps = [(feat_dict[imp], imp) for imp in feat_sorted]
        output['feat_imps'] = feat_imps

    return output


if __name__ == '__main__':
    from data import get_data
    X, y = get_data()
    print(train_model(X, y, return_feature_importances = False))
