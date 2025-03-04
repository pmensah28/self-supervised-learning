import torch
import torch.nn as nn

class SupervisedTrainer:
    """
    A class to handle supervised training and evaluation of a model.
    """
    def __init__(self, model, device, train_loader, test_loader, lr=0.05, momentum=0.9, weight_decay=1e-5):
        """
        Initialize the trainer with model, data loaders, and optimizer parameters.
        
        :param model: The neural network model
        :param device: The computing device (CPU or GPU)
        :param train_loader: DataLoader for training data
        :param test_loader: DataLoader for testing data
        :param lr: Learning rate for optimization
        :param momentum: Momentum for SGD optimizer
        :param weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    
    def train(self, epochs):
        """
        Train the model using supervised learning.
        
        :param epochs: Number of training epochs
        """
        self.model.train()
        torch.set_grad_enabled(True)
        
        for e in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += output.shape[0] * loss.item()
            print(f"Epoch {e}: {epoch_loss / len(self.train_loader.dataset):.4f}")
    
    def test(self):
        """
        Evaluate the model on the test set.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.0f}%)\n')




# import torch
# import torch.nn as nn

# def train_supervised(model, device, train_loader, epoch):
#     model.train()
#     torch.set_grad_enabled(True)

#     optimizer = torch.optim.SGD(
#         filter(lambda x: x.requires_grad, model.parameters()),
#         lr=0.05,
#         momentum=0.9,
#         weight_decay=10**(-5)
#     )

#     criterion = nn.CrossEntropyLoss().to(device)

#     for e in range(epoch):
#         epoch_loss = 0.0
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += output.shape[0] * loss.item()
#         print(f"Epoch {e}: {epoch_loss / len(train_loader.dataset)}")

# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)
#     print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
