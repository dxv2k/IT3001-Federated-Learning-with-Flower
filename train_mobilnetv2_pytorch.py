import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import torchvision
import torchvision.transforms as transforms

# Check if CUDA (GPU) is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the model (MobileNetV2)
class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2, self).__init__()
        self.model = torchvision.models.mobilenet_v2(
            pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


model = MobileNetV2(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False
)

training_metrics = []
validation_metrics = []
epoch_times = []


def fit(parameters, config):
    model.load_state_dict(parameters)

    for epoch in range(config["epochs"]):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        training_metrics.append({"loss": total_loss / len(train_loader)})

        model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)

        validation_loss = total_loss / len(test_loader)
        validation_accuracy = total_correct / total_samples
        validation_metrics.append(
            {
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        print(
            f"Epoch {epoch + 1}/{config['epochs']} - "
            f"Train Loss: {training_metrics[-1]['loss']:.4f} - "
            f"Validation Loss: {validation_loss:.4f} - "
            f"Validation Accuracy: {validation_accuracy:.4f} - "
            f"Time: {epoch_time:.2f} seconds"
        )

    # Save metrics to CSV
    metrics_df = pd.DataFrame(
        {
            "train_loss": [item["loss"] for item in training_metrics],
            "validation_loss": [
                item["validation_loss"] for item in validation_metrics
            ],
            "validation_accuracy": [
                item["validation_accuracy"] for item in validation_metrics
            ],
            "epoch_time": epoch_times,
        }
    )
    metrics_df.to_csv("training_metrics.csv", index=False)

    # Save model weights
    torch.save(model.state_dict(), "model_weights.pth")


# Example usage
config = {"epochs": 50}  # Change this to the desired number of epochs
initial_weights = model.state_dict()
fit(initial_weights, config)
