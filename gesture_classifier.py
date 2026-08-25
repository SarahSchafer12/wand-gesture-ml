import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ── 1. LOAD DATA ────────────────────────────────────────────────────────────
# Put your JSONs in folders: data/still/, data/left/, data/right/

def load_recording(filepath):
    with open(filepath) as f:
        data = json.load(f)
    window = np.array([[s['ax'], s['ay'], s['az']] for s in data['sensors']])
    return window

def load_dataset(data_dir):
    data_dir = Path(data_dir)
    X, y = [], []
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        label = class_dir.name  # folder name = class label
        for json_file in class_dir.glob("*.json"):
            window = load_recording(json_file)
            features = extract_features(window)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# ── 2. FEATURE EXTRACTION ───────────────────────────────────────────────────

def extract_features(window):
    """
    window: np.array of shape (N, 3) — columns are ax, ay, az
    Returns a flat feature vector.
    """
    features = []

    # Per-axis stats
    for i, name in enumerate(['ax', 'ay', 'az']):
        axis = window[:, i]
        features += [
            np.mean(axis),
            np.std(axis),
            np.min(axis),
            np.max(axis),
            np.max(axis) - np.min(axis),        # range
            np.sum(np.abs(np.diff(axis))),       # total variation
            np.mean(np.abs(axis - np.mean(axis))), # mean absolute deviation
        ]

    # Magnitude stats
    mag = np.sqrt(np.sum(window**2, axis=1))
    features += [
        np.mean(mag),
        np.std(mag),
        np.max(mag),
        np.max(mag) - np.min(mag),
        np.sum(np.abs(np.diff(mag))),
    ]

    # Directional: net displacement (useful for left vs right)
    features += [
        window[-1, 0] - window[0, 0],  # net ax change
        window[-1, 1] - window[0, 1],  # net ay change
        window[-1, 2] - window[0, 2],  # net az change
    ]

    # Peak detection: where does max magnitude occur (normalized position)
    features.append(np.argmax(mag) / len(mag))

    return features

# ── 3. TRAIN ────────────────────────────────────────────────────────────────

def train(data_dir):
    print("Loading data...")
    X, y = load_dataset(data_dir)
    print(f"  {len(X)} recordings, classes: {np.unique(y)}")

    clf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Leave-One-Out CV — best choice with small datasets (10 per class)
    print("\nRunning Leave-One-Out cross-validation...")
    loo = LeaveOneOut()
    scores = cross_val_score(clf, X, y, cv=loo, scoring='accuracy')
    print(f"  LOO Accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    # Full predictions for confusion matrix
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(clf, X, y, cv=loo)
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    labels = sorted(np.unique(y))
    print("Labels:", labels)
    print(confusion_matrix(y, y_pred, labels=labels))

    # Train final model on all data
    print("\nTraining final model on all data...")
    clf.fit(X, y)

    # Feature importance
    feature_names = []
    for name in ['ax', 'ay', 'az']:
        feature_names += [f'{name}_mean', f'{name}_std', f'{name}_min',
                          f'{name}_max', f'{name}_range', f'{name}_totalvar', f'{name}_mad']
    feature_names += ['mag_mean', 'mag_std', 'mag_max', 'mag_range', 'mag_totalvar']
    feature_names += ['net_ax', 'net_ay', 'net_az', 'peak_position']

    print("\nTop 10 most important features:")
    importances = sorted(zip(clf.feature_importances_, feature_names), reverse=True)
    for imp, name in importances[:10]:
        print(f"  {name:20s} {imp:.3f}")

    return clf

# ── 4. SAVE / LOAD ──────────────────────────────────────────────────────────

def save_model(clf, path="gesture_model.pkl"):
    joblib.dump(clf, path)
    print(f"\nModel saved to {path}")

def load_model(path="gesture_model.pkl"):
    return joblib.load(path)

# ── 5. LIVE INFERENCE (for your UDP loop) ───────────────────────────────────

def predict_gesture(clf, window_samples):
    """
    window_samples: list of dicts like [{"ax": 12, "ay": -1, "az": 224}, ...]
    Returns predicted label string.
    """
    window = np.array([[s['ax'], s['ay'], s['az']] for s in window_samples])
    features = extract_features(window).reshape(1, -1)
    return clf.predict(features)[0]

# ── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    clf = train(data_dir)
    save_model(clf)
