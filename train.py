from catboost import CatBoostClassifier, CatBoostRegressor

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


def fit_regression_model():
    model = CatBoostRegressor(iterations=10,
                              learning_rate=0.2,
                              depth=3,
                              verbose=0)
    model.fit(train_data, train_labels, cat_features, embedding_features=embedding_features)
    # Get predictions
    preds = model.predict(eval_data)
    print(preds)  # [0.46018641 0.47496323 0.65977057]
    model.save_model('regression.bin')


def fit_classification_model():
    model = CatBoostClassifier(iterations=10,
                               learning_rate=0.2,
                               depth=3,
                               verbose=0)
    model.fit(train_data, train_labels, cat_features, embedding_features=embedding_features)
    # Get predictions
    preds = model.predict_proba(eval_data)
    print(preds)  # [0.46018641 0.47496323 0.65977057]
    model.save_model('classifier.bin')


fit_regression_model()
fit_classification_model()

#  [0.46018641 0.47496323 0.65977057]
# [[0.56015706 0.43984294]
#  [0.55445946 0.44554054]
#  [0.43797584 0.56202416]]
