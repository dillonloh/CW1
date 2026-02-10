from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import zipfile
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm.auto import tqdm

# DILLON: for loading features
from torch.utils.data import DataLoader

# DILLON: seed for reproducibility
torch.manual_seed(69)
np.random.seed(69)

BASE_DATA_PATH = Path(".") / "data"
ZIP_PATH = BASE_DATA_PATH / "part1.zip"
DATA_PATH = BASE_DATA_PATH / "part1"


FRUIT_CLASS_NAMES = [
    "granny_smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple",
    "banana",
    "jackfruit",
    "custard_apple",
    "pomegranate",
]

datas = {}


def download_data():
    BASE_DATA_PATH.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        file_id = "1qNFjQIBck90I41aiJMpZR70NZRHQtEqE"
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            output=str(ZIP_PATH),
            quiet=True,
        )

    if not DATA_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(str(BASE_DATA_PATH))


def load_data(split_name: str):
    download_data()
    images = np.load(DATA_PATH / f"{split_name}_images.npy")
    labels = np.load(DATA_PATH / f"{split_name}_labels.npy")
    return images, labels


def get_data(split_name: str):
    if split_name not in datas:
        datas[split_name] = load_data(split_name)
    return datas[split_name]


class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, split_name: str, transform: Optional[Callable] = None):
        self.images, self.labels = get_data(split_name)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self):
        return int(np.max(self.labels) + 1)


def visualize_samples(
    split_name: str,
    num_rows: int = 2,
    figsize: tuple[int, int] = (10, 5),
    seed: int = 0,
):
    dataset = FruitDataset(split_name)
    rng = np.random.RandomState(seed)
    _, axes = plt.subplots(num_rows, dataset.num_classes // num_rows, figsize=figsize)
    for class_idx in range(dataset.num_classes):
        row = class_idx // (dataset.num_classes // num_rows)
        col = class_idx % (dataset.num_classes // num_rows)
        class_indices = np.where(dataset.labels == class_idx)[0]
        random_idx = int(rng.choice(class_indices))
        img, _ = dataset[random_idx]
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{FRUIT_CLASS_NAMES[class_idx]}")
        axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(f"part1_{split_name}_samples.png")
    plt.close()


def create_feature_extractor(model_name: str, pretrained_cfg: str, device: str = "cuda:0"):
    model = timm.create_model(model_name, pretrained=True, pretrained_cfg=pretrained_cfg)
    model.transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.head = nn.Identity()
    return model.eval().to(device)


def get_features(
    split_name: str,
    feature_extractor: torch.nn.Module,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda:0",
):
    dataset = FruitDataset(split_name, transform=feature_extractor.transform)
    feature_extractor.eval()
    features = None

    ### YOUR CODE STARTS HERE ###

    batch_features = []

    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        for x, y in dataloader:
            batch_features.append(feature_extractor(x.to(device)).cpu())
            print(f"Extracted features for {len(batch_features)*batch_size} / {len(dataset)} samples", end="\r")
        features = torch.cat(batch_features).cpu().numpy()
        
    ### YOUR CODE ENDS HERE ###
    
    return features, dataset.labels, dataset.num_classes


class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, num_classes: int):
        assert features.ndim == 2, f"Expected (N, D), got {features.shape}"
        self.features = features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.num_classes = int(num_classes)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.features[index]).float()
        y = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.labels)

    @classmethod
    def create(cls, split_name: str, feature_extractor: torch.nn.Module, **kwargs):
        features, labels, num_classes = get_features(split_name, feature_extractor, **kwargs)
        return cls(features, labels, num_classes)


def visualize_features_tsne(
    features_dataset,
    title: str = "Features t-SNE",
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 0,
):
    from sklearn.manifold import TSNE

    tsne = TSNE(perplexity=perplexity, max_iter=n_iter, random_state=random_state)
    features_2d = tsne.fit_transform(features_dataset.features)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=features_dataset.labels,
        cmap="tab10",
        alpha=0.7,
    )
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=FRUIT_CLASS_NAMES,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(f"part1_{title.replace(' ', '_').lower()}.png")
    plt.close()


class MultinomialLogisticRegression(nn.Module):
    def __init__(self, n_input_features, n_classes):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=n_input_features, out_features=n_classes)

    def forward(self, x):

        # single linear feed forward layer
        x = self.linear(x)
        output = torch.softmax(x, dim=1)

        return output


def train_linear_probe(
    features_dataset: FeaturesDataset,
    num_epochs: int = 32,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    num_workers: int = 16,
    device: str = "cuda:0",
):
    linear_probe = None
    epoch_losses = []

    ### YOUR CODE STARTS HERE ###

    print(f"Training on {device}")

    n_input_features = features_dataset.features.shape[1]
    n_classes = features_dataset.labels.max(axis=0) + 1

    linear_probe = MultinomialLogisticRegression(n_input_features=n_input_features, n_classes=n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(linear_probe.parameters(), lr=learning_rate, weight_decay=weight_decay)

    dataloader = DataLoader(features_dataset, batch_size=batch_size, num_workers=num_workers)

    for n in range(num_epochs):
        running_loss = 0
        for i, (x, y) in enumerate(dataloader):
            yhat = linear_probe.forward(x.to(device))
            loss = loss_fn(yhat, y.to(device))
            running_loss += loss.item()
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            print(f"Epoch {n} / {num_epochs} | Batch {i} / {len(dataloader)} | Loss = {loss.item()}", end="\r")

        epoch_loss = (running_loss/len(dataloader)).cpu()
        print(f"Epoch {n} complete | Loss = {epoch_loss}")
        
        epoch_losses.append(epoch_loss)

    ### YOUR CODE ENDS HERE ###

    return linear_probe, epoch_losses

def evaluate_linear(
    linear_probe: torch.nn.Module,
    val_feats: FeaturesDataset,
    device: str = "cuda:0",
):
    accuracy = 0.0

    ### YOUR CODE STARTS HERE ###
    
    dataloader = DataLoader(val_feats, batch_size=100, num_workers=4)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            yhat = linear_probe.forward(x.to(device)).cpu()
            y_predicted = torch.argmax(yhat, dim=1)
            y_correct = (y_predicted == y)
            correct += y_correct.sum().item()
            total += y_predicted.shape[0] 

    accuracy = correct/total
    print(f"Correct: {correct} / {total}")
    ### YOUR CODE ENDS HERE ###

    return accuracy
def train_finetune_probe(
    split_name: str,
    feature_extractor: torch.nn.Module,
    pretrained_linear_probe: torch.nn.Module = None,
    num_epochs: int = 2,
    batch_size: int = 32,
    feature_lr: float = 1e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device: str = "cuda:0",
):
    ### YOUR CODE STARTS HERE ###
    ### YOUR CODE ENDS HERE ###

    return model, epoch_losses




def plot_losses(losses, name):
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training loss for {name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"part1_{name}_training_loss.png")
    plt.show()
    plt.close()