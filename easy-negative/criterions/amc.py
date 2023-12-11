import torch
import torch.nn.functional as F
def Amc(device, features):
    #import pdb; pdb.set_trace()
    bs = features.shape[0]/2

    labels = torch.cat([torch.arange(bs) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    #print(features.shape)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # logits = torch.cat([positives, negatives], dim=1)

    # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

    # logits = logits / self.args.temperature

    m = 0.5
    negatives = torch.clamp(negatives,min=-1+1e-7,max=1-1e-7)
    clip = torch.acos(negatives)
    b1 = m - clip
    mask = b1>0
    l1 = torch.sum((mask*b1)**2)
    positives = torch.clamp(positives,min = -1+1e-7,max = 1-1e-7)
    l2 = torch.acos(positives)
    l2 = torch.sum(l2**2)

    loss = (l1 + l2)/25

    return loss
