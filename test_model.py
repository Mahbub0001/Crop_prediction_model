import pickle
import numpy as np
import os

p = 'model.pkl'
if not os.path.exists(p):
    print('model.pkl not found')
    raise SystemExit(1)

with open(p, 'rb') as f:
    model = pickle.load(f)

print('Model type:', type(model))
try:
    print('Model classes_:', getattr(model, 'classes_', None))
except Exception as e:
    print('Could not read classes_', e)
try:
    print('n_features_in_:', getattr(model, 'n_features_in_', None))
    print('feature_importances_:', getattr(model, 'feature_importances_', None))
except Exception as e:
    print('Could not read feature info:', e)

samples = [
    [90,42,43,20.88,82.00,6.50,202.93],
    [10,10,10,30,50,5,10],
    [50,40,40,25,60,7,100],
    [0,0,0,0,0,0,0]
]
arr = np.array(samples)
print('Samples array dtype:', arr.dtype)

for i, s in enumerate(samples):
    pred = model.predict(np.array([s]))
    print(f'sample {i} -> raw predict: {pred} (type: {type(pred)})')
    try:
        print('  as list:', list(pred))
        print('  as scalar repr:', repr(np.asarray(pred).ravel()[0]))
    except Exception as e:
        print('  error showing pred details:', e)

# Also test using pandas DataFrame
import pandas as pd
cols=['N','P','K','temperature','humidity','ph','rainfall']
df = pd.DataFrame(samples, columns=cols)
print('\nPredicting on DataFrame:')
try:
    preds_df = model.predict(df)
    print('DataFrame predict:', preds_df)
except Exception as e:
    print('DataFrame predict raised:', e)

print('Done')

# Quick random sweep to see prediction distribution
import collections
rng = np.random.default_rng(42)
rand_samples = np.column_stack([
    rng.integers(0, 141, size=200),  # N
    rng.integers(0, 146, size=200),  # P
    rng.integers(0, 206, size=200),  # K
    rng.uniform(8, 45, size=200),    # temp
    rng.uniform(14, 100, size=200),  # humidity
    rng.uniform(3.5, 9.5, size=200), # ph
    rng.uniform(20, 300, size=200)   # rainfall
])
preds = model.predict(rand_samples)
cnt = collections.Counter(preds.tolist())
print('\nRandom sweep counts (label:count):')
for k, v in sorted(cnt.items()):
    print(k, v)

# Recreate label encoder mapping from dataset
try:
    df_all = pd.read_csv('Crop_recommendation.csv')
    labels = df_all['label'].astype(str).values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(labels)
    inv_map = {i: lbl for i, lbl in enumerate(le.classes_)}
    print('\nLabelEncoder classes_ (index:label):')
    for i, lbl in inv_map.items():
        print(i, lbl)
    # Show mapping for most common predicted label
    most_common = cnt.most_common(1)[0][0]
    print('\nMost common predicted numeric label', most_common, 'maps to', inv_map.get(most_common))
except Exception as e:
    print('Could not recreate label mapping from CSV:', e)
