"""
Trains and compares candidate digit classifiers on data/digits (see
generate_digit_crops.py), then reports accuracy, model size, and
per-image inference latency for each so the choice is made on measured
evidence, not assumption.

Candidates:
  - HOG + Linear SVM       (scikit-learn, no GPU, tiny model)
  - HOG + RandomForest     (scikit-learn, no GPU, tiny model)
  - Small CNN              (PyTorch - already a dependency via EasyOCR;
                             exported to ONNX for lightweight deployment
                             if it wins, so the final choice doesn't need
                             full PyTorch installed at inference time)
"""
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog

import sys
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/digits")
CLASSES = list("0123456789")


def load_split(split: str):
    X_img, y = [], []
    for digit in CLASSES:
        for f in (DATA_DIR / split / digit).glob("*.png"):
            X_img.append(np.array(Image.open(f)))
            y.append(int(digit))
    return np.array(X_img), np.array(y)


def extract_hog(images: np.ndarray) -> np.ndarray:
    feats = [
        hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        for img in images
    ]
    return np.array(feats)


def model_size_bytes(obj) -> int:
    buf = io.BytesIO()
    import pickle
    pickle.dump(obj, buf)
    return buf.tell()


def bench_sklearn(name, clf, X_train, y_train, X_test, y_test, imgs_test):
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    preds = clf.predict(X_test)
    acc = (preds == y_test).mean()

    # per-image inference latency (feature extraction + predict)
    t0 = time.time()
    for img in imgs_test[:200]:
        f = extract_hog(img[None, ...])
        clf.predict(f)
    latency_ms = (time.time() - t0) / min(200, len(imgs_test)) * 1000

    size_kb = model_size_bytes(clf) / 1024
    print(f"{name:20s}  acc={acc*100:5.1f}%  train={train_time:5.1f}s  "
          f"latency={latency_ms:5.2f}ms/img  size={size_kb:7.1f}KB")
    return clf, acc


def train_cnn(X_train_img, y_train, X_val_img, y_val, X_test_img, y_test):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyDigitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 16 * 16, 32)
            self.fc2 = nn.Linear(32, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))   # 64 -> 32
            x = self.pool(F.relu(self.conv2(x)))   # 32 -> 16
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    def to_tensor(imgs):
        t = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1) / 255.0
        return t

    X_train_t = to_tensor(X_train_img)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = to_tensor(X_val_img)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = to_tensor(X_test_img)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = TinyDigitCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.time()
    best_val_acc = 0.0
    best_state = None
    batch_size = 64
    n = len(X_train_t)
    for epoch in range(15):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(X_train_t[idx])
            loss = F.cross_entropy(out, y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).argmax(dim=1)
        acc = (test_preds == y_test_t).float().mean().item()

    t0 = time.time()
    with torch.no_grad():
        for i in range(min(200, len(X_test_t))):
            model(X_test_t[i:i+1])
    latency_ms = (time.time() - t0) / min(200, len(X_test_t)) * 1000

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_kb = buf.tell() / 1024

    print(f"{'Tiny CNN':20s}  acc={acc*100:5.1f}%  train={train_time:5.1f}s  "
          f"latency={latency_ms:5.2f}ms/img  size={size_kb:7.1f}KB")
    return model, acc


def main():
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    print("Loading splits...")
    X_train_img, y_train = load_split("train")
    X_val_img, y_val = load_split("val")
    X_test_img, y_test = load_split("test")
    print(f"train={len(X_train_img)}  val={len(X_val_img)}  test={len(X_test_img)}")

    print("Extracting HOG features...")
    X_train_hog = extract_hog(X_train_img)
    X_test_hog = extract_hog(X_test_img)

    print()
    print(f"{'Model':20s}  {'Accuracy':>9s}  {'TrainTime':>10s}  {'Latency':>12s}  {'Size':>10s}")
    print("-" * 75)

    results = {}
    svm, acc = bench_sklearn("HOG + Linear SVM", SVC(kernel="linear", C=1.0, class_weight="balanced"),
                              X_train_hog, y_train, X_test_hog, y_test, X_test_img)
    results["svm"] = (svm, acc)

    rf, acc = bench_sklearn("HOG + RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
                             X_train_hog, y_train, X_test_hog, y_test, X_test_img)
    results["rf"] = (rf, acc)

    cnn, acc = train_cnn(X_train_img, y_train, X_val_img, y_val, X_test_img, y_test)
    results["cnn"] = (cnn, acc)

    print()
    print("In-distribution results tied or nearly tied - saving all three for")
    print("a harder out-of-distribution stress test (see stress_test_digit_classifier.py).")

    Path("data/models").mkdir(parents=True, exist_ok=True)
    import joblib
    import torch
    joblib.dump(results["svm"][0], "data/models/digit_classifier_svm.joblib")
    joblib.dump(results["rf"][0], "data/models/digit_classifier_rf.joblib")
    torch.save(results["cnn"][0].state_dict(), "data/models/digit_classifier_cnn.pt")
    print("Saved all 3 candidates to data/models/")


if __name__ == "__main__":
    main()
