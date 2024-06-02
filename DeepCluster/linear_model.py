import copy
import torch
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils import compute_features

def linear_model(model_base, train_loader, test_loader, device):
    model = copy.deepcopy(model_base).to(device)
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    features, labels = compute_features(train_loader, model, len(train_loader.dataset), get_labels=True)

    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, max_iter=10000))
    clf.fit(features, labels)

    x_test, y_true = [], []

    torch.set_grad_enabled(False)
    for pics, labels in test_loader:
        pics = pics.to(device)
        model.eval()
        features_test = model(pics)
        x_test.append(features_test.cpu().numpy())
        y_true.append(labels)

    x_test = np.concatenate(x_test)
    y_true = np.concatenate(y_true)
    y_pred = clf.predict(x_test)
    print(f"Test Accuracy: {accuracy_score(y_true, y_pred)}")
