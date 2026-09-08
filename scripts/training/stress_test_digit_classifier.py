"""
Stress-tests the three saved digit classifier candidates (data/models/) on
harder, wider-range perturbations than generate_digit_crops.py's training
augmentation used - rotation, blur, noise, and contrast pushed further out
of distribution, plus low-resolution simulation (small on-card digits).

The in-distribution test set tied all 3 candidates at 100% - not a real
signal, just an easy benchmark. This is the real tiebreaker: which one
generalizes best when conditions are worse than training saw.
"""
import random
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from skimage.feature import hog

WESTERN_TO_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
FONT_PATH = "C:/Windows/Fonts/tahomabd.ttf"
CANVAS = 64


class TinyDigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def render_clean_digit(digit: str, font_size: int) -> Image.Image:
    glyph = digit.translate(WESTERN_TO_ARABIC_INDIC)
    font = ImageFont.truetype(FONT_PATH, font_size)
    img = Image.new("L", (CANVAS, CANVAS), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), glyph, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (CANVAS - w) // 2 - bbox[0]
    y = (CANVAS - h) // 2 - bbox[1]
    draw.text((x, y), glyph, font=font, fill=0)
    return img


def hard_augment(img: Image.Image, rng: random.Random) -> Image.Image:
    # Wider rotation range than training (+/-6 -> +/-15)
    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)

    # Wider scale jitter, including simulating a small/undersized crop
    scale = rng.uniform(0.6, 1.3)
    new_size = max(8, int(CANVAS * scale))
    img = img.resize((new_size, new_size), Image.LANCZOS)
    canvas = Image.new("L", (CANVAS, CANVAS), color=255)
    off = ((CANVAS - new_size) // 2, (CANVAS - new_size) // 2)
    if new_size <= CANVAS:
        canvas.paste(img, off)
    else:
        crop_off = (-off[0], -off[1])
        canvas = img.crop((crop_off[0], crop_off[1], crop_off[0] + CANVAS, crop_off[1] + CANVAS))
    img = canvas

    # Heavier blur (out-of-focus phone photo)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 2.5)))

    arr = np.array(img).astype(np.float32)

    # More extreme contrast/brightness (harsh lighting, glare)
    contrast = rng.uniform(0.5, 1.6)
    brightness = rng.uniform(-45, 45)
    arr = (arr - 128) * contrast + 128 + brightness
    arr = np.clip(arr, 0, 255)

    # Heavier noise
    noise = np.random.normal(0, rng.uniform(10, 30), arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    img = Image.fromarray(arr.astype(np.uint8))

    # Simulate low-res capture then upscale back (real phone photos of a
    # small printed field are often effectively low-resolution)
    if rng.random() < 0.5:
        small = img.resize((rng.randint(16, 32),) * 2, Image.BILINEAR)
        img = small.resize((CANVAS, CANVAS), Image.BILINEAR)

    return img


def generate_stress_set(n_per_class: int, seed: int):
    rng = random.Random(seed)
    np.random.seed(seed)
    X, y = [], []
    for digit in "0123456789":
        for _ in range(n_per_class):
            font_size = int(44 * rng.uniform(0.85, 1.15))
            clean = render_clean_digit(digit, font_size)
            aug = hard_augment(clean, rng)
            X.append(np.array(aug))
            y.append(int(digit))
    return np.array(X), np.array(y)


def eval_sklearn(name, path, X_test_img, y_test):
    clf = joblib.load(path)
    feats = np.array([
        hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        for img in X_test_img
    ])
    t0 = time.time()
    preds = clf.predict(feats)
    latency_ms = (time.time() - t0) / len(X_test_img) * 1000
    acc = (preds == y_test).mean()
    print(f"{name:20s}  acc={acc*100:5.1f}%  latency={latency_ms:6.3f}ms/img")
    return acc


def eval_cnn(X_test_img, y_test):
    model = TinyDigitCNN()
    model.load_state_dict(torch.load("data/models/digit_classifier_cnn.pt"))
    model.eval()
    X_t = torch.tensor(X_test_img, dtype=torch.float32).unsqueeze(1) / 255.0
    t0 = time.time()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).numpy()
    latency_ms = (time.time() - t0) / len(X_test_img) * 1000
    acc = (preds == y_test).mean()
    print(f"{'Tiny CNN':20s}  acc={acc*100:5.1f}%  latency={latency_ms:6.3f}ms/img")
    return acc


def main():
    print("Generating out-of-distribution stress test set (harder than training augmentation)...")
    X_test, y_test = generate_stress_set(n_per_class=150, seed=999)
    print(f"Stress test set: {len(X_test)} images (150/class)")
    print()

    results = {}
    results["svm"] = eval_sklearn("HOG + Linear SVM", "data/models/digit_classifier_svm.joblib", X_test, y_test)
    results["rf"] = eval_sklearn("HOG + RandomForest", "data/models/digit_classifier_rf.joblib", X_test, y_test)
    results["cnn"] = eval_cnn(X_test, y_test)

    print()
    winner = max(results.items(), key=lambda kv: kv[1])
    print(f"Winner under stress test: {winner[0]}  ({winner[1]*100:.1f}%)")


if __name__ == "__main__":
    main()
