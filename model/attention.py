import torch
import torch.nn as nn
import torch.nn.functional as F
import time
class AdditiveAttention(nn.Module):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(AdditiveAttention, self).__init__()

        if internal_dim is None:
            internal_dim = int((encoder_hidden_state_dim + decoder_hidden_state_dim) / 2)

        self.w1 = nn.Linear(encoder_hidden_state_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(decoder_hidden_state_dim, internal_dim, bias=False)
        self.v = nn.Linear(internal_dim, 1, bias=False)

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        # return value should be of shape (batch, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + self.w2(decoder_state)))

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_obj, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        score_vec = torch.cat([self.score(encoder_states[:, i], decoder_state) for i in range(encoder_states.shape[1])],
                              dim=1)
        # score_vec is of shape (batch, num_obj)

        attention_probs = torch.unsqueeze(F.softmax(score_vec, dim=1), dim=2)
        # attention_probs is of shape (batch, num_obj, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, enc_dim)

        return final_context_vec, attention_probs


class TemporallyBatchedAdditiveAttention(AdditiveAttention):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(TemporallyBatchedAdditiveAttention, self).__init__(encoder_hidden_state_dim,
                                                                 decoder_hidden_state_dim,
                                                                 internal_dim)

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, num_enc_states, max_time, enc_dim)
        # decoder_state is of shape (batch, max_time, dec_dim)
        # return value should be of shape (batch, num_enc_states, max_time, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + torch.unsqueeze(self.w2(decoder_state), dim=1)))

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, max_time, enc_dim)
        # decoder_state is of shape (batch, max_time, dec_dim)
        score_vec = self.score(encoder_states, decoder_state)
        # score_vec is of shape (batch, num_enc_states, max_time, 1)

        attention_probs = F.softmax(score_vec, dim=1)
        # attention_probs is of shape (batch, num_enc_states, max_time, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, max_time, enc_dim)

        return final_context_vec, torch.squeeze(torch.transpose(attention_probs, 1, 2), dim=3)

class MapEncoder(nn.Module):
    def __init__(self, out_channels, backbone='resnet18'):
        super().__init__()
        # self.backbone = torch.nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     BasicBlock(64,64),
        #     BasicBlock(64,64)
        # )
        # resnet18 = models.resnet18(pretrained=True)
        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # efficientnet_b0 = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = torch.nn.Sequential(*list(resnet18.children())[:-2])
        self.out_channels = out_channels
        with torch.no_grad():
            n_flatten = self.backbone(torch.zeros((2,3,50,50)).float()).shape[1]
        self.fc = nn.Linear(n_flatten, out_channels)

    def forward(self, route):
        x = F.adaptive_avg_pool2d(self.backbone(route),(1,1)).flatten(1)
        return self.fc(x)

class SimpleMapEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros((1,3,25,25)).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, out_channels), nn.ReLU())
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out