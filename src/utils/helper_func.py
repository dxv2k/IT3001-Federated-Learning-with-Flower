import os
import csv
import time

import numpy as np
import torch


def get_parameters(net) -> list[np.ndarray]:
    return [p.cpu().detach().numpy() for p in net.parameters()]


def set_parameters(net, parameters: list[np.ndarray]) -> None:
    new_parameters = [torch.tensor(p, dtype=torch.float32) for p in parameters]
    for current_param, new_param in zip(net.parameters(), new_parameters):
        current_param.data = new_param


def train(net, trainloader, epochs: int, round: int, device: torch.device) -> list:
    """Train the network on the training set."""
    # print("Start train ...")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    net.to(device)
    net.train()

    metrics_list = []
    for epoch in range(epochs):
        start_time = time.time()

        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_time = time.time() - start_time
        epoch_loss /= len(trainloader)
        epoch_loss = epoch_loss.detach().item()
        # epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
        metrics_list.append([epoch, round, epoch_loss, epoch_acc, 0.0, 0.0, epoch_time])
        # TODO: save weights here 

    return metrics_list


def test(net, testloader, round: int, device: torch.device):
    """Evaluate the network on the entire test set."""
    # print("Start test ...")

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    net.to(device)
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(testloader)
    # loss /= len(testloader.dataset)
    accuracy = correct / total

    metrics_list = [[0, round, 0.0, 0.0, loss, accuracy, 0.0]]
    return loss, accuracy, metrics_list


def save_metrics_to_csv(filename, metrics_list):
    if not os.path.exists("metrics"):
        os.makedirs("metrics")
    file_path = os.path.join("metrics", filename)

    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "epoch",
                    "round",
                    "train_loss",
                    "train_accuracy",
                    "validation_loss",
                    "validation_accuracy",
                    "epoch_time",
                ]
            )

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(metrics_list)
