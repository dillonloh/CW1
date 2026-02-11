import numpy as np
import matplotlib.pyplot as plt

# DILLON : added
import os
import pickle
from architectures import FinetuneProbeModel, MultinomialLogisticRegression
from utils import save_model, load_model

from representations import (
    create_feature_extractor,
    visualize_samples,
    FeaturesDataset,
    visualize_features_tsne,
    train_linear_probe,
    train_finetune_probe,
    evaluate_linear,
    evaluate_finetune,
    plot_losses,
)

device = "cuda:0"

feature_extractors = {
    "clip": create_feature_extractor("vit_base_patch16_clip_224", "openai", device=device),
    "dino": create_feature_extractor("vit_base_patch16_224", "dino", device=device),
    "mae":  create_feature_extractor("vit_base_patch16_224", "mae", device=device),
}

# Visualise some samples
visualize_samples("photo_val", seed=0)

# DILLON: check to see if i already pickled the dataset features from previous runs
# Extract features for training set
os.makedirs("features_datasets", exist_ok=True)
if os.path.exists(os.path.join("features_datasets", "train_features_datasets.pkl")):
    print("Features previously extracted")
    with open(os.path.join("features_datasets", "train_features_datasets.pkl"), "rb") as f:
        train_features_datasets = pickle.load(f)

else:
    train_features_datasets = {}
    for method_name, fx in feature_extractors.items():
        print(f"Extracting features for training set using {method_name}...")
        train_features_datasets[method_name] = FeaturesDataset.create(
            "photo_train", fx, device=device
        )
        assert train_features_datasets[method_name].features.shape[0] == 13000
        assert train_features_datasets[method_name].features.ndim == 2

# DILLON: pickle the datasets so i dont need to keep redoing this
with open(os.path.join("features_datasets", "train_features_datasets.pkl"), "wb") as f:
    pickle.dump(train_features_datasets, f)

# # Visualize features using t-SNE
# for method_name in feature_extractors.keys():
#     print(f"Visualizing features for {method_name}...")
#     visualize_features_tsne(
#         train_features_datasets[method_name],
#         title=f"Features t-SNE ({method_name})"
#     )

# DILLON: Check if training linear probes have already been pickled
os.makedirs("models", exist_ok=True)
os.makedirs("losses", exist_ok=True)

linear_probes = {}

for method_name in feature_extractors.keys():
    model_path = os.path.join("models", f"linear_probe_{method_name}.pth")
    if os.path.exists(model_path):
        print(f"Loading linear probe for {method_name}...")
        
        probe = MultinomialLogisticRegression(
            n_input_features=train_features_datasets[method_name].features.shape[1],
            n_classes=train_features_datasets[method_name].num_classes,
        ).to(device)

        load_model(probe, model_path, map_location=device)
        linear_probes[method_name] = probe

    else: 
        # Linear probe training
        print(f"Training linear probe for {method_name}...")
        linear_probes[method_name], losses, batch_losses = train_linear_probe(
            train_features_datasets[method_name], device=device
        )
        plot_losses(losses, method_name)
        plot_losses(batch_losses, f"{method_name}_batch")

        with open(os.path.join("losses", f"{method_name}_linear_probe_epoch_losses.pkl"), "wb") as f:
            pickle.dump(losses, f)

        with open(os.path.join("losses", f"{method_name}_linear_probe_batch_losses.pkl"), "wb") as f:
            pickle.dump(batch_losses, f)

        # DILLON: save the models so i dont need to keep redoing this
        save_model(
            linear_probes[method_name],
            model_path,
            extra={"method": method_name},
        )

# Evaluate linear probes on photo_val (implement evaluate_linear from scratch)
for method_name, probe in linear_probes.items():
    val_feats = FeaturesDataset.create("photo_val", feature_extractors[method_name], device=device)
    acc = evaluate_linear(probe, val_feats, device=device)
    print(f"{method_name} linear probe photo-val accuracy: {acc:.4f}")

# DILLON: Check if trained finetune probes have already been pickled
os.makedirs("models", exist_ok=True)
os.makedirs("losses", exist_ok=True)

# Finetune probe training (init from linear probe)
finetuned_models = {}
for method_name, fx in feature_extractors.items():
    model_path = os.path.join("models", f"finetuned_{method_name}.pt")

    if os.path.exists(model_path):
        print(f"Loading finetuned {method_name} model...")

        model = FinetuneProbeModel(
            feature_extractor=fx,
            linear_probe=linear_probes[method_name],
        ).to(device)

        load_model(model, model_path, map_location=device)
        finetuned_models[method_name] = model

    else:
        print(f"Fine-tuning {method_name} (init from linear probe)...")
        finetuned_models[method_name], ft_losses, batch_losses = train_finetune_probe(
            "photo_train",
            fx,
            pretrained_linear_probe=linear_probes[method_name],
            device=device,
        )

        plot_losses(ft_losses, f"{method_name}_finetune")
        plot_losses(batch_losses, f"{method_name}_finetune_batch")
    
        save_model(
            finetuned_models[method_name],
            model_path,
            extra={"method": method_name},
        )

        with open(os.path.join("losses", f"finetuned_{method_name}_epoch_losses.pkl"), "wb") as f:
            pickle.dump(ft_losses, f)

        with open(os.path.join("losses", f"finetuned_{method_name}_batch_losses.pkl"), "wb") as f:
            pickle.dump(batch_losses, f)

        
# Evaluate finetuned models on photo_val (implement evaluate_finetune from scratch)
for method_name, fx in feature_extractors.items():
    print(f"Evaluating {method_name} finetuned model...")
    acc = evaluate_finetune(finetuned_models[method_name], "photo_val", fx, device=device)
    print(f"{method_name} finetune photo-val accuracy: {acc:.4f}")
