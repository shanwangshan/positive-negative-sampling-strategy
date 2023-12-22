import torch

class ClassificationWrapper(torch.nn.Module):
    def __init__(self, feature_extractor, n_classes, feat_name, feat_dim, pooling_op=None, use_dropout=False, dropout=0.5):
        super(ClassificationWrapper, self).__init__()
        self.feature_extractor = feature_extractor
        self.feat_name = feat_name
        self.use_dropout = use_dropout
        if pooling_op is not None:
            self.pooling = eval('torch.nn.'+pooling_op)
        else:
            self.pooling = None
        if use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(feat_dim, n_classes)

    def forward(self, *inputs):
        emb = self.feature_extractor(*inputs, return_embs=True)[self.feat_name]
        emb_pool = self.pooling(emb) if self.pooling is not None else emb
        emb_pool = emb_pool.view(inputs[0].shape[0], -1)
        if self.use_dropout:
            emb_pool = self.dropout(emb_pool)
        logit = self.classifier(emb_pool)
        return logit
