"""Entry point for running the AngelNet MNIST demo."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from angelnet_core import AngelNet


def build_dataloader(data_dir: Path, batch_size: int) -> torch.utils.data.DataLoader:
    """Create a data loader for the MNIST training split."""
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root=str(data_dir), train=True, transform=transform, download=True)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def train(
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    data_dir: Path,
    storage_dir: Path | None = None,
    device: torch.device | None = None,
) -> None:
    """Run the training loop for the AngelNet demo."""
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {resolved_device}")

    input_size = 784
    hidden_size = 128
    num_classes = 10

    dataloader = build_dataloader(data_dir, batch_size)

    net = AngelNet(
        input_dim=input_size,
        hidden_dim=hidden_size,
        output_dim=num_classes,
        base_lr=learning_rate,
        storage_dir=storage_dir,
    ).to(resolved_device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(net.parameters()) + list(net.transformer.parameters()), lr=learning_rate)
    net.set_optimizer(optimizer)

    total_time = 0.0
    total_batches = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"Starting Epoch {epoch + 1}")

        correct_total = 0
        total_samples = 0

        for batch_index, (images, labels) in enumerate(dataloader):
            batch_start = time.time()
            images = images.to(resolved_device)
            labels = labels.to(resolved_device)

            outputs = net(images, labels, data_type="image")

            if batch_index == 0:
                decoded_data = net.transformer.decode(outputs[0], data_type="image")
                print(f"[Main] Decoded vector norm: {decoded_data.norm().item():.4f}")

            if net.training_mode:
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / batch_size
                print(
                    f"[Debug] Batch {total_batches + 1}: Correct predictions: {correct}/{batch_size}, "
                    f"Accuracy: {accuracy:.2%}"
                )
                net.think(
                    images,
                    labels,
                    accuracy=accuracy,
                    success=accuracy > net.goal.target_accuracy,
                    loss=loss.item(),
                    data_type="image",
                )
            else:
                predicted = net.autonomous_classify(images, data_type="image")
                correct = (predicted == labels).sum().item()
                accuracy = correct / batch_size
                print(
                    f"[Debug] Batch {total_batches + 1}: Correct predictions: {correct}/{batch_size}, "
                    f"Accuracy: {accuracy:.2%}"
                )
                net.think(
                    images,
                    labels,
                    accuracy=accuracy,
                    success=accuracy > net.goal.target_accuracy,
                    loss=0.0,
                    data_type="image",
                )

            correct_total += correct
            total_samples += batch_size

            batch_time = time.time() - batch_start
            print(f"Batch {total_batches + 1} took {batch_time:.4f} seconds")
            total_batches += 1

        epoch_accuracy = correct_total / total_samples
        print(f"Epoch {epoch + 1} accuracy: {epoch_accuracy:.2%}")

        net.save_memory()
        net.visualize_metrics(epoch + 1)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1} took {epoch_time:.4f} seconds")
        total_time += epoch_time

    print(f"Total time: {total_time:.4f} seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AngelNet demo on MNIST")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for Adam")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory where MNIST data will be stored",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=None,
        help="Directory where AngelNet persistent state will be stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage_dir = args.storage_dir or os.getenv("ANGELNET_STORAGE_DIR")
    resolved_storage = Path(storage_dir) if storage_dir is not None else None
    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
        storage_dir=resolved_storage,
    )


if __name__ == "__main__":
    main()
