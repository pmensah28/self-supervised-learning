import torch
from torch.utils.tensorboard import SummaryWriter
from torchlars import LARS
from models import SimCLRModel
from data import get_data_loaders
from train import train

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-3
    temperature = 0.1
    train_loader = get_data_loaders(batch_size)
    model = SimCLRModel().to(device)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6)
    optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
    writer = SummaryWriter()
    train(model, device, train_loader, optimizer, num_epochs, temperature, writer)

if __name__ == '__main__':
    main()