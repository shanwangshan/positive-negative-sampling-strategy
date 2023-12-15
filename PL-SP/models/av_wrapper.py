import random
import torch
import torch.nn as nn


# __all__ = [
#     'av_wrapper'
# ]


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


class AV_Wrapper(nn.Module):
    def __init__(self, video_model, audio_model, proj_dim=128):
        super(AV_Wrapper, self).__init__()
        self.video_model = video_model
        self.audio_model = audio_model


        self.use_linear_proj = proj_dim is not None
        if proj_dim is not None:
            self.video_proj = Head(video_model.out_dim, proj_dim)
            self.audio_proj = Head(audio_model.out_dim, proj_dim)
            self.out_dim = self.video_proj.out_dim
        else:
            self.out_dim = video_model.out_dim

    def forward(self, video, audio):


        video_emb = self.video_model(video)
        video_emb = video_emb.view(video_emb.shape[0], video_emb.shape[1])
        if self.use_linear_proj:
            video_emb = self.video_proj(video_emb)

        audio_emb = self.audio_model(audio)
        audio_emb = audio_emb.view(audio_emb.shape[0], audio_emb.shape[1])
        if self.use_linear_proj:
            audio_emb = self.audio_proj(audio_emb)

        return video_emb, audio_emb


def av_wrapper(proj_dim=128):
    import models

    from models.audio import Conv2D
    from models.video import R2Plus1D

    video_model = R2Plus1D()
    audio_model = Conv2D(depth=10)
    model = AV_Wrapper(video_model, audio_model, proj_dim=proj_dim)

    return model


def main():
    import sys


    sys.path.insert(0, '.')
    from criterions.contrastive import ContrastiveLoss
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = av_wrapper(proj_dim=None)
    model.cuda()
    model.train()
    print(model)


    # Dummy data
    dummy_video = torch.ones((4, 3, 8, 112, 112)).cuda()
    dummy_audio = torch.ones((4, 1, 100, 128)).cuda()

    video_emb, audio_emb = model(dummy_video, dummy_audio)
    infonce = ContrastiveLoss(input_dim=512,proj_dim=[128],target='cross-modal',temperature=0.07, normalize=True,device=device)
    loss = infonce(video_emb,audio_emb)


if __name__ == '__main__':
    main()
