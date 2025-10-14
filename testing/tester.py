import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


class Tester:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.classes = self.test_loader.dataset.classes

    def test(self):
        y_pred = []
        y_true = []
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        return y_true, y_pred

    def analyze(self, y_true, y_pred):
        logging.info("Classification Report:")
        logging.info(
            "\n" + classification_report(y_true, y_pred, target_names=self.classes)
        )

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.classes,
            yticklabels=self.classes,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

        self._visualize_predictions()

    def _visualize_predictions(self, num_images=25):
        # Get a batch of test data
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        images_for_viz = images.cpu().numpy()
        images, labels = images.to(self.device), labels.to(self.device)

        # Get predictions
        outputs = self.model(images)
        _, predicted = torch.max(outputs, 1)

        # Plot images with predictions
        fig = plt.figure(figsize=(15, 7))
        for idx in np.arange(num_images):
            if idx >= len(images):
                break
            ax = fig.add_subplot(
                (num_images // 5) + 1, 5, idx + 1, xticks=[], yticks=[]
            )
            # Un-normalize
            img = images_for_viz[idx] / 2 + 0.5
            plt.imshow(np.transpose(img, (1, 2, 0)))
            ax.set_title(
                f"{self.classes[predicted[idx].item()]} ({self.classes[labels[idx].item()]})",
                color=("green" if predicted[idx] == labels[idx] else "red"),
            )
        plt.show()
