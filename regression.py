from catboost import CatBoostRegressor

# Initialize data

cat_features = [0]

train_data = [['a', 1, 4, 5, 6, ],
              ['b', 4, 5, 6, 7, ],
              ['c', 30, 40, 50, 60]]

eval_data = [['a', 2, 4, 6, 8, ],
             ['b', 1, 4, 50, 60, ]]

train_targets = [10, 20, 30]
# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=10,
                          learning_rate=1,
                          depth=2)
# Fit model
model.fit(train_data, train_targets, cat_features)
# Get predictions
preds = model.predict(eval_data)
print(preds)  # [15.65772339 20.38869995]
model.save_model('regression.bin')
