 Transaction fraud detection model

 Task description
The model classifies financial transactions as legitimate (`isFraud = 0`) or fraudulent (`isFraud = 1`) based on numerical transaction features.

 Data used
Source: PaySim synthetic dataset (mobile transaction simulation).
https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download

 Features
- `step` — transaction time (1 step = 1 hour).
- `amount` — transaction amount.
- `oldbalanceOrg` — sender's balance before the transaction.
- `newbalanceOrig` — sender's balance after the transaction.
- `oldbalanceDest` — recipient's balance before the transaction.
- `newbalanceDest` — recipient's balance after the transaction.

 Target variable
- `isFraud` — class label (0 = normal, 1 = fraud).

 Model selection
Model: `DecisionTreeClassifier` (scikit-learn).

Why this one?
1. Interpretability — a decision tree allows you to understand the rules by which a decision is made (important for fintech).
2. Working with unbalanced data — the `class_weight` parameters and depth restrictions help reduce overfitting.
3. Training and prediction speed — critical for real-time systems.
4. No need to scale features — trees do not require normalization.

 Model parameters
- `max_depth=10` — tree depth limit.
- `min_samples_split=5` — minimum number of objects for splitting.
- `min_samples_leaf=2` — minimum number of objects in a leaf.
- `random_state=42` — reproducibility of results.

 Evaluation results
- Accuracy: 0.9987 (99.87%)
- Precision (Fraud): 0.95 — the proportion of correctly identified fraudulent transactions among all those marked as fraud.
- Recall (Fraud): 0.89 — the proportion of detected fraudulent transactions out of all actual fraud.
- F1-score (Fraud): 0.92 — the balance between precision and recall.

Error matrix:
[[49950 0]
[ 134 16]]

 Using the model
1. Training: run the `train.py` script.
2. Saving: the model is saved to the `fraud_detection_model.pkl` file.
3. Prediction: use `model.predict(X)` for new data.

Dependencies
- python >= 3.8
- pandas
- numpy
- scikit-learn
- joblib

