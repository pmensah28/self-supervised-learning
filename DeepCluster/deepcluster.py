import numpy as np
import faiss
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from utils import compute_features, AverageMeter, UnifLabelSampler, cluster_assign

def DeepCluster(model, device, train_loader, epoch, k):
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    model = model.to(device)

    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=0.05,
        momentum=0.9,
        weight_decay=10**(-5)
    )

    criterion = nn.CrossEntropyLoss().to(device)

    for e in range(epoch):
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        features = compute_features(train_loader, model, len(train_loader.dataset))

        pipeline = Pipeline([('scaling', StandardScaler())])
        post_scale = pipeline.fit_transform(features)
        post_norm = normalize(post_scale, norm="l2")

        d = post_norm.shape[1]
        clus = faiss.Clustering(d, k)
        clus.seed = np.random.randint(1234)
        clus.niter = 20
        clus.max_points_per_centroid = 60000

        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        clus.train(post_norm, index)
        _, I = index.search(post_norm, 1)
        labels = np.squeeze(I)

        unique, counts = np.unique(labels, return_counts=True)
        print(f"Epoch {e}: Cluster assignments: {dict(zip(unique, counts))}")

        images_lists = [[] for _ in range(k)]
        for i in range(len(train_loader.dataset)):
            images_lists[labels[i]].append(i)

        train_dataset = cluster_assign(images_lists, train_loader.dataset)
        sampler = UnifLabelSampler(int(1 * len(train_dataset)), images_lists)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=4, sampler=sampler)

        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, k)
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        torch.set_grad_enabled(True)
        loss = train(train_dataloader, model, criterion, optimizer, e)
        print(loss.cpu().numpy())

def train(loader, model, crit, opt, epoch):
    losses = AverageMeter()
    model.train()
    optimizer_tl = torch.optim.SGD(model.top_layer.parameters(), lr=0.01, weight_decay=10**-5)

    for i, (input_tensor, target) in enumerate(loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = crit(output, target_var)
        losses.update(loss.data, input_tensor.size(0))

        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

    return losses.avg
