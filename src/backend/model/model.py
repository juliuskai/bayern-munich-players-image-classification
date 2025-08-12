from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from src.backend.preprocessing.dataset_builder import get_dataloaders


class PlayerClassifier:
    def __init__(self, data_dir, num_classes=5, batch_size=32, lr=1e-3, model_save_path='trained-models/resnet18-players.pth'):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model().to(self.device)
        self.train_loader, self.val_loader, self.class_names, self.train_targets = get_dataloaders(data_dir, batch_size=self.batch_size)

    def compute_weights(self):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_targets),
            y=self.train_targets
        )
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        return weights_tensor

    def _build_model(self):
        """
        load pretrained ResNet18 and replace and custom make final layer.
        """
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def train(self, num_epochs=10):
        weights = self.compute_weights()
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(self.model.fc.parameters(), lr=self.lr)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}")

        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"model saved to {self.model_save_path}")

    def evaluate(self):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                y_true.extend(labels.tolist())
                y_pred.extend(preds.cpu().tolist())

        print("\nEvaluation Report:\n")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        plot_confusion_matrix(self.model, self.val_loader, self.class_names, self.device)

    def load_trained_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        self.model.eval()
        print("trained model loaded.")


def plot_confusion_matrix(model, val_loader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize by row

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()