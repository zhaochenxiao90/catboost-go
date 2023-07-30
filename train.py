from catboost import CatBoostClassifier


# a classification model with numerical, categorical and embedding features
def fit_full_feature_classification_model():
    # Initialize data
    cat_features = [3]
    embedding_features = [0, 1]
    train_data = [
        [[0.1, 0.12, 0.33], [1.0, 0.7], 2, "male"],
        [[0.0, 0.8, 0.2], [1.1, 0.2], 1, "female"],
        [[0.2, 0.31, 0.1], [0.3, 0.11], 2, "female"],
        [[0.01, 0.2, 0.9], [0.62, 0.12], 1, "male"],
    ]

    train_labels = [1, 0, 0, 1]
    eval_data = [
        [[0.2, 0.1, 0.3], [1.2, 0.3], 1, "female"],
        [[0.33, 0.22, 0.4], [0.98, 0.5], 2, "female"],
        [[0.78, 0.29, 0.67], [0.76, 0.34], 2, "male"],
    ]
    model = CatBoostClassifier(iterations=10, learning_rate=0.2, depth=3, verbose=0)
    model.fit(train_data, train_labels, cat_features=cat_features, embedding_features=embedding_features)
    model.save_model('full_features.bin')
    # Get predictions
    print('full_features')
    preds = model.predict_proba(eval_data)
    print(preds)  # [0.46018641 0.47496323 0.65977057]
    # [[0.56015706 0.43984294]
    #  [0.55445946 0.44554054]
    #  [0.43797584 0.56202416]]


fit_full_feature_classification_model()


def fit_numerical_classification_model():
    train_data = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
    ]
    train_labels = [1, 0, 0, 1]
    eval_data = [
        [2, 4, 6],
        [3, 5, 7],
        [4, 6, 8],
    ]
    model = CatBoostClassifier(iterations=10, learning_rate=0.2, depth=3, verbose=0)
    model.fit(train_data, train_labels, cat_features=[], embedding_features=[])
    print('numerical_only')
    model.save_model('numerical_only.bin')
    # Get predictions
    preds = model.predict_proba(eval_data)
    print(preds)


fit_numerical_classification_model()


def fit_category_classification_model():
    train_data = [
        ["a", "a", "b"],
        ["b", "b", "b"],
        ["b", "b", "a"],
        ["a", "a", "a"],
    ]
    train_labels = [1, 0, 0, 1]
    eval_data = [
        ["a", "a", "a"],
        ["a", "a", "b"],
        ["b", "b", "a"],
        ["b", "b", "b"],
    ]
    model = CatBoostClassifier(iterations=10, learning_rate=0.2, depth=3, verbose=0)
    model.fit(train_data, train_labels, cat_features=[0, 1, 2], embedding_features=[])
    print('category_only')
    model.save_model('category_only.bin')
    # Get predictions
    preds = model.predict_proba(eval_data)
    print(preds)


fit_category_classification_model()


def fit_embedding_classification_model():
    # Initialize data
    train_data = [
        [[0.1, 0.12, 0.33], [1.0, 0.7]],
        [[0.0, 0.8, 0.2], [1.1, 0.2]],
        [[0.2, 0.31, 0.1], [0.3, 0.11]],
        [[0.01, 0.2, 0.9], [0.62, 0.12]],
    ]

    train_labels = [1, 0, 0, 1]
    eval_data = [
        [[0.2, 0.1, 0.3], [1.2, 0.3]],
        [[0.33, 0.22, 0.4], [0.98, 0.5]],
        [[0.78, 0.29, 0.67], [0.76, 0.34]],
    ]
    model = CatBoostClassifier(iterations=10, learning_rate=0.2, depth=3, verbose=0)
    model.fit(train_data, train_labels, cat_features=[], embedding_features=[0, 1])
    model.save_model('embedding_only.bin')
    # Get predictions
    print('embedding_only')
    preds = model.predict_proba(eval_data)
    print(preds)


fit_embedding_classification_model()

# full_features
# [[0.56015706 0.43984294]
#  [0.55445946 0.44554054]
# [0.43797584 0.56202416]]
# numerical_only
# [[0.5149358  0.4850642 ]
#  [0.44320543 0.55679457]
# [0.42863304 0.57136696]]
# category_only
# [[0.38610862 0.61389138]
#  [0.38610862 0.61389138]
# [0.61389138 0.38610862]
# [0.61389138 0.38610862]]
# embedding_only
# [[0.53241853 0.46758147]
#  [0.53961316 0.46038684]
# [0.53961316 0.46038684]]