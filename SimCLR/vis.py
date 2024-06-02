import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Visualization with t-SNE
def visualize_tsne(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for i in range(10): # Assuming there are 10 classes in CIFAR-10
        indices = labels == i
        plt.scatter(embeddings_tsne[indices, 0], embeddings_tsne[indices, 1], label=f'Class {i}', alpha=0.5)
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show()
