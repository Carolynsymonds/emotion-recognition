import json
import os
import matplotlib.pyplot as plt


class MetricsLogger:
    def __init__(self):
        self.metrics = {}  # Dynamic dictionary to store any metric

    def log_epoch(self, **kwargs):
        """
        Store any metric as a key-value pair for one epoch.
        Example:
        logger.log_epoch(train_loss=0.5, val_loss=0.6, accuracy=0.8)
        """
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_metrics_history(self):
        return self.metrics

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f)
        print(f"Metrics saved to {filepath}")

    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.metrics = json.load(f)
            print(f"Metrics loaded from {filepath}")
        else:
            print(f"No file found at {filepath}")

    def plot(self, keys=None):
        keys = keys or self.metrics.keys()
        for key in keys:
            plt.plot(self.metrics[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.title("Training Metrics")
        plt.show()
