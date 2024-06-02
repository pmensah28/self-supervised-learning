
import torch
from models import SimpleCnn
from train import train_supervised
from DeepCluster.data import get_data_loaders
from deepcluster import DeepCluster
from linear_model import linear_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader_unsupervised, train_loader_supervised, test_loader = get_data_loaders()
    simpleCNN = SimpleCnn().to(device)
    DeepCluster(simpleCNN, device, train_loader_unsupervised, 5, 10)
    random_CNN = SimpleCnn()
    trainCNN = SimpleCnn().to(device)
    train_supervised(trainCNN, device, train_loader_unsupervised, 5)
    linear_model(random_CNN, train_loader_supervised, test_loader, device)
    linear_model(simpleCNN, train_loader_supervised, test_loader, device)
    linear_model(trainCNN, train_loader_supervised, test_loader, device)

if __name__ == '__main__':
    main()




