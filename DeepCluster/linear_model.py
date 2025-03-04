import copy
import torch
import numpy as np
from torch import nn
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils import compute_features

class LinearModelTrainer:
    """
    A class to train and evaluate a linear classifier on extracted features from a model.
    """
    def __init__(self, model_base, train_loader, test_loader, device):
        """
        Initialize the Linear Model Trainer.
        
        :param model_base: The base neural network model
        :param train_loader: DataLoader for training data
        :param test_loader: DataLoader for testing data
        :param device: The computing device (CPU or GPU)
        """
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Clone and modify the base model
        self.model = copy.deepcopy(model_base).to(device)
        self.model.top_layer = None
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])

    def train_and_evaluate(self):
        """
        Train a linear classifier on extracted features and evaluate on the test set.
        """
        # Compute features and labels for training
        features, labels = compute_features(self.train_loader, self.model, len(self.train_loader.dataset), get_labels=True)
        
        # Train a linear SVM classifier
        clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, max_iter=10000))
        clf.fit(features, labels)
        
        # Compute test features
        x_test, y_true = self._extract_test_features()
        
        # Make predictions and evaluate accuracy
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy}")

    def _extract_test_features(self):
        """
        Extract features from the test set using the trained model.
        
        :return: Tuple of (test features, true labels)
        """
        x_test, y_true = [], []
        
        torch.set_grad_enabled(False)
        for pics, labels in self.test_loader:
            pics = pics.to(self.device)
            self.model.eval()
            features_test = self.model(pics)
            x_test.append(features_test.cpu().numpy())
            y_true.append(labels.numpy())
        
        return np.concatenate(x_test), np.concatenate(y_true)





# import copy
# import torch
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# from utils import compute_features

# def linear_model(model_base, train_loader, test_loader, device):
#     model = copy.deepcopy(model_base).to(device)
#     model.top_layer = None
#     model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
#     features, labels = compute_features(train_loader, model, len(train_loader.dataset), get_labels=True)

#     clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, max_iter=10000))
#     clf.fit(features, labels)

#     x_test, y_true = [], []

#     torch.set_grad_enabled(False)
#     for pics, labels in test_loader:
#         pics = pics.to(device)
#         model.eval()
#         features_test = model(pics)
#         x_test.append(features_test.cpu().numpy())
#         y_true.append(labels)

#     x_test = np.concatenate(x_test)
#     y_true = np.concatenate(y_true)
#     y_pred = clf.predict(x_test)
#     print(f"Test Accuracy: {accuracy_score(y_true, y_pred)}")
