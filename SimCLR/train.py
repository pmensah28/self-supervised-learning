import torch
from tqdm import tqdm
from losses import nt_xent_loss
from vis import visualize_tsne
import numpy as np

def train(model, device, train_loader, optimizer, num_epochs, temperature, writer):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        all_embeddings = []
        all_labels = []

        for images, labels in tqdm(train_loader):
            # Get two augmented views of the same image
            img1, img2 = images.to(device), images.to(device)

            z1, p1 = model(img1)
            z2, p2 = model(img2)

            loss = nt_xent_loss(p1, p2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            all_embeddings.append(z1.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())

        # Concatenate all embeddings and  labels to make access to specific indices easier ( for visualization) 
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        writer.add_scalar('Loss', total_loss / len(train_loader), epoch)

        # Visualize embeddings with t-SNE
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            visualize_tsne(embeddings, labels)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

    writer.close()
