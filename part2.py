import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import math
import numpy as np
import einops
from pathlib import Path
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from DIT import DiT

# DILLON: for saving model and losses
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_SIZE = 32
PATCH_SIZE = 4
DIM = 256
DEPTH = 4
HEADS = 4
MLP_RATIO = 4.0
CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 40
LR = 3e-4
T = 500
device = "cuda"

# DILLON: added for part (b) conditional generation
conditional = True

# DILLON: seed for reproducibility
torch.manual_seed(69)
np.random.seed(69)

# ============================================================================
# DDPM UTILS
# ============================================================================


def get_ddpm_schedule(T):
    betas = torch.linspace(1e-4, 0.03, T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def forward_diffusion(x0, t, alphas_cumprod):
    ### YOUR CODE STARTS HERE ###
    # Your code should compute xt at timestep t, and also return the noise
    
    mean_xt = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1) * x0
    var_xt = (1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
    noise = torch.randn(mean_xt.shape).to(device=device)
    scaled_noise = torch.sqrt(var_xt) * noise
    xt = mean_xt + scaled_noise

    ### YOUR CODE ENDS HERE ###
    return xt, noise


@torch.no_grad()
def sample_ddpm(net, T, bsz, betas, alphas, alphas_cumprod, num_snapshots=10): # reverse process
    net.eval()

    # sample the initial noise
    x = torch.randn(bsz, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)

    # Identify which timesteps to save for the visualization grid
    snapshot_indices = torch.linspace(T - 1, 0, num_snapshots).long()
    snapshots = []
    
    if conditional: # DILLON: modified for part (b) conditional generation
        labels = torch.randint(0, 10, (bsz,), device=device)

    for t in reversed(range(T)):

        ### YOUR CODE STARTS HERE ###
        # Your code should compute xt at timestep t
        
        a = 1 / torch.sqrt(alphas[t]).view(-1, 1, 1, 1)
        b = (1 - alphas[t]) / (torch.sqrt(1 - alphas_cumprod[t])).view(-1, 1, 1, 1)

        tensor_t = torch.tensor([t], device=device) # need be tensor otherwise DIT will complain about device not being attribute
        if conditional:
            c = net(x, tensor_t, labels) # DILLON: modified for part (b) conditional generation
        else:
            c = net(x, tensor_t)

        d = torch.sqrt(betas[t]).view(-1, 1, 1, 1) # according to DDPM paper, this is one option for sigma_t
        
        if t == 0:
            z = torch.zeros_like(x)
        else:
            z = torch.randn_like(x)
        
        x = a * (x - (b * c)) + (d * z)
        
        ### YOUR CODE ENDS HERE ###

        if t in snapshot_indices:
            snapshots.append(x.cpu())

    # Return shape: (num_snapshots, bsz, C, H, W)
    return torch.stack(snapshots)


def visualize_forward_diffusion(dataloader, alphas_cumprod, n_steps=10):
    # Get a batch of real images
    images, _ = next(iter(dataloader))
    images = images[:8]

    # Select timesteps to show (0 to T-1)
    indices = torch.linspace(0, T - 1, n_steps).long()

    cols = []
    for t in indices:
        t_batch = torch.full((images.shape[0],), t, dtype=torch.long)
        # Apply the forward diffusion
        xt, _ = forward_diffusion(images.to(device), t_batch.to(device), alphas_cumprod)
        cols.append(xt.cpu())

    # Stack and rearrange: (Steps, Batch, C, H, W) -> (Batch * Steps, C, H, W)
    result = torch.stack(cols, dim=1)
    result = einops.rearrange(result, "b t c h w -> (b t) c h w")

    grid = vutils.make_grid(result, nrow=n_steps, normalize=True, value_range=(-1, 1))

    plt.figure(figsize=(15, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    if conditional:
        plt.savefig("conditional_images/forward_diffusion_process_conditional.png")
    else:
        plt.savefig("images/forward_diffusion_process.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    Path("images").mkdir(exist_ok=True)
    Path("conditional_images").mkdir(exist_ok=True)

    # 1. Data Setup
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    # 2. Model Setup
    net = DiT(
        input_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=CHANNELS,
        hidden_size=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        mlp_ratio=MLP_RATIO,
        num_classes=(0 if not conditional else 10), # 10 because 0 - 9 digits
        learn_sigma=False,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")

    optimizer = optim.AdamW(net.parameters(), lr=LR)
    # get noise schedule parameters
    betas, alphas, alphas_cumprod = get_ddpm_schedule(T)

    # Visualize the forward process before training
    visualize_forward_diffusion(dataloader, alphas_cumprod)
    
    model_path = os.path.join("part2_models", "dit_mnist_final_conditional.pth" if conditional else "dit_mnist_final.pth")

    # DILLON: do this so i dont have to keep retraining
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        net.load_state_dict(torch.load(model_path, map_location=device))
        loss_history = np.load(os.path.join("part2_models", "loss_history_conditional.npy" if conditional else "loss_history.npy")).tolist()
        print("Model loaded.")

    else:
        # 3. Training Loop
        print("Starting Training...")
        loss_history = []

        print(f"Training {'conditional' if conditional else 'unconditional'} DiT Model...")
        
        for epoch in range(EPOCHS):
            net.train()
            epoch_loss = 0
            for i, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                bsz = images.shape[0]

                # Sample random timesteps
                t = torch.randint(0, T, (bsz,), device=device).long()

                # Add noise
                xt, noise = forward_diffusion(images, t, alphas_cumprod)

                # Predict noise
                if conditional:
                    # DILLON: modified for part (b) conditional generation
                    pred_noise = net(xt, t, labels.to(device))
                
                else:
                    pred_noise = net(xt, t)

                # Use MSE loss
                loss = nn.functional.mse_loss(pred_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

            # Inside the training loop:
            if (epoch + 5) % 5 == 0:
                num_samples = 8
                num_steps = 10  # How many steps of the process to show

                # Generate the trajectory
                # Shape: (num_steps, num_samples, C, H, W)
                traj = sample_ddpm(
                    net, T, num_samples, betas, alphas, alphas_cumprod, num_steps
                )

                grid_ready = einops.rearrange(traj, "s b c h w -> b s c h w")
                grid_ready = einops.rearrange(grid_ready, "b s c h w -> (b s) c h w")

                # Create the grid
                grid = vutils.make_grid(
                    grid_ready, nrow=num_steps, normalize=True, value_range=(-1, 1)
                )

                # Save the result
                if conditional:
                    vutils.save_image(grid, f"conditional_images/evolution_epoch_{epoch+1}_conditional.png")
                    print(
                        f"Epoch {epoch+1}: Saved generation grid to conditional_images/evolution_epoch_{epoch+1}_conditional.png"
                    )
                else:
                    vutils.save_image(grid, f"images/evolution_epoch_{epoch+1}.png")    
                    print(
                        f"Epoch {epoch+1}: Saved generation grid to images/evolution_epoch_{epoch+1}.png"
                    )

        # DILLON: save the final model and losses
        os.makedirs("part2_models", exist_ok=True)
        if conditional:
            torch.save(net.state_dict(), os.path.join("part2_models", "dit_mnist_final_conditional.pth"))
            np.save(os.path.join("part2_models", "loss_history_conditional.npy"), np.array(loss_history))
        else:
            torch.save(net.state_dict(), os.path.join("part2_models", "dit_mnist_final.pth"))
            np.save(os.path.join("part2_models", "loss_history.npy"), np.array(loss_history))
        
    # 4. Final Sampling & Visualization
    print("Generating Final Samples...")
    trajectory = sample_ddpm(net, T, 16, betas, alphas, alphas_cumprod)

    # Save the final result
    final_grid = vutils.make_grid(
        trajectory[-1], nrow=4, normalize=True, value_range=(-1, 1)
    )
    if conditional:
        vutils.save_image(final_grid, "conditional_images/dit_mnist_final_conditional.png")
    else:
        vutils.save_image(final_grid, "images/dit_mnist_final.png")

    # Save Loss Plot
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")

    if conditional:
        plt.savefig("conditional_images/loss_curve_conditional.png")
        print("Done! Check the 'conditional_images' folder.")

    else:
        plt.savefig("images/loss_curve.png")
        print("Done! Check the 'images' folder.")
