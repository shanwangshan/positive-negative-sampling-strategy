import torch
from torch import nn
import torch.nn.functional as F
from criterions.infonce import InfoNCELoss



class Head(nn.Module):
    def __init__(self, input_dim, proj_dims):
        super(Head, self).__init__()

        if not isinstance(proj_dims, list):
            proj_dims = [proj_dims]

        projection = []

        for i, d in enumerate(proj_dims):
            projection += [nn.Linear(input_dim, d)]
            input_dim = d
            if i < len(proj_dims)-1:
                projection += [nn.ReLU(inplace=True)]
        self.projection = nn.Sequential(*projection)
        self.out_dim = proj_dims[-1]

    def forward(self, x):
        return self.projection(x)


class ContrastiveLoss(nn.Module):
    def __init__(self,
                 input_dim=512,
                 proj_dim=None,
                 target='cross-modal',
                 temperature=0.07,
                 normalize=True,device=None):
        super(ContrastiveLoss, self).__init__()
        self.video_projection = Head(input_dim, proj_dim) if proj_dim is not None else None
        self.video_projection = self.video_projection.to(device)
        self.audio_projection = Head(input_dim, proj_dim) if proj_dim is not None else None
        self.audio_projection = self.audio_projection.to(device)
        assert target in {'cross-modal', 'within-modal'}
        self.target = target
        assert temperature > 0.
        self.temperature = temperature
        assert isinstance(normalize, bool)
        self.normalize = normalize
        self.contrastive_loss = InfoNCELoss(temperature=temperature, normalize=normalize)

    # def predict(self, video_emb, audio_emb):
    #     import pdb; pdb.set_trace()
    #     video_emb = self.video_projection(video_emb)
    #     audio_emb = self.audio_projection(audio_emb)
    #     return video_emb, audio_emb

    def forward(self, video_emb, audio_emb, *args):
        losses = {}

        losses['V2A'] = self.contrastive_loss(
            video_emb, audio_emb, choices_dim=0, output_head=self.video_projection, target_head=self.audio_projection)
        losses['A2V'] = self.contrastive_loss(
            audio_emb, video_emb, choices_dim=0, output_head=self.audio_projection, target_head=self.video_projection)



        total_loss = sum([losses[k] for k in losses]) / float(len(losses))


        return total_loss
