import numpy as np

import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


SMALL = 1e-08


class UpperBound(nn.Module, ABC):
    @abstractmethod
    def update(self, y_samples):

        raise NotImplementedError


class UpperWithPosterior(UpperBound):
    """
    后验分布 p(y|x) 均假设为高斯分布
    """
    def __init__(
        self,
        embedding_dim=768,
        hidden_dim=500,
        tag_dim=128,
        device=None
    ):
        super(UpperWithPosterior, self).__init__()
        self.device = device
        # u
        self.p_mu = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                  # nn.ReLU(),
                                  nn.Tanh(),
                                  nn.Linear(hidden_dim, tag_dim))
        # log(σ**2)
        self.p_log_var = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                       nn.Tanh(),
                                       # nn.ReLU(),
                                       nn.Linear(hidden_dim, tag_dim),
                                       # nn.Tanh()
                                       )

    # 返回 u , log(σ**2)
    def get_mu_logvar(self, embeds):
        mean = self.p_mu(embeds)
        log_var = self.p_log_var(embeds)

        return mean, log_var

    def loglikeli(self, y_samples, mu, log_var):
        # [batch, seq_len, dim]
        return (-0.5 * (mu - y_samples) ** 2 / log_var.exp()
                + log_var
                + torch.log(math.pi)
                ).sum(dim=1).mean(dim=0)

    # 从正态分布中 sample 样本
    def get_sample_from_param_batch(self, mean, log_var, sample_size):
        bsz, seqlen, tag_dim = mean.shape
        z = torch.randn(bsz, sample_size, seqlen, tag_dim).to(self.device)

        z = z * torch.exp(0.5 * log_var).unsqueeze(1).expand(-1, sample_size, -1, -1) + \
            mean.unsqueeze(1).expand(-1, sample_size, -1, -1)

        # [batch * sample_size, seq_len, tag_dim]
        return z.view(-1, seqlen, tag_dim)

    @abstractmethod
    def update(self, y_samples):
        raise NotImplementedError


class VIB(UpperWithPosterior):
    """
    Deep Variational Information Bottleneck
    """
    # TODO
    # 表示该高斯分布与 N（0，1）之间的KL散度
    def update(self, x_samples):  # [nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)

        return 1. / 2. * (mu ** 2 + logvar.exp() - 1. - logvar).mean()


class CLUB(UpperWithPosterior):
    """
    CLUB: Mutual Information Contrastive Learning Upper Bound
    """

    def mi_est_sample(self, y_samples, mu, log_var):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        positive = - (mu - y_samples) ** 2 / 2. / log_var.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / 2. / log_var.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound/2.
        return upper_bound

    def update(self, x_samples):
        """
        f(x_sample) = u, var -> sampling -> y_sample
        return mi_est_sample(x_sample, y_sample)

        :param x_samples:
        :return:
        """
        mu, log_var = self.get_mu_logvar(x_samples)
        y_samples = self.get_sample_from_param_batch(mu, log_var, 1).squeeze(0)

        return self.mi_est_sample(y_samples, mu, log_var)


class vCLUB(UpperBound):
    """
     vCLUB: Variational Mutual Information Contrastive Learning Upper Bound
    """
    def __init__(self):
        super(vCLUB, self).__init__()

    def mi_est_sample(self, y_samples):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        return self.mi_est(y_samples, y_samples[random_index])

    def mi_est(self, y_samples, y_n_samples):
        """
        approximate q(y|x) - q(y'|x)

        :param y_samples: [-1, 1, hidden_dim]
        :param y_n_samples: [-1, 1, hidden_dim]
        :return:
               mi estimation [nsample, 1]
        """
        positive = torch.zeros_like(y_samples)
        #
        negative = - (y_samples - y_n_samples) ** 2 / 2.
        # TODO mean 分母为 seq_len*batch, 最后数值过小
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound/2.
        return upper_bound

    def mse(self, y_samples, y_n_samples):
        """
        approximate q(y|x) - q(y'|x)

        :param y_samples: [-1, 1, hidden_dim]
        :param y_n_samples: [-1, 1, hidden_dim]
        :return:
               mi estimation [nsample, 1]
        """

        return (y_samples - y_n_samples) ** 2 / 2

    def consine(self, y_samples, y_n_samples):
        """
        approximate q(y|x) - q(y'|x)

        :param y_samples: [-1, 1, hidden_dim]
        :param y_n_samples: [-1, 1, hidden_dim]
        :return:
               mi estimation [nsample, 1]
        """
        return torch.cosine_similarity(y_samples, y_n_samples, dim=-1)

    def loglikeli(self, x_samples, y_samples):
        return 0

    def update(self, y_samples, y_n_samples):

        return self.mi_est(y_samples, y_n_samples)


# TODO
class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, device=None):
        super(InfoNCE, self).__init__()
        self.lower_size = 100
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, self.lower_size),
                                    nn.ReLU(),
                                    nn.Linear(self.lower_size, 1),
                                    nn.Softplus())
        self.device = device

    def set_com_score_fun(self, x_dim, y_dim, max_seq_len):
        for i in range(max_seq_len):
            F_func = nn.Sequential(nn.Linear(x_dim + y_dim, self.lower_size),
                                   nn.ReLU(),
                                   nn.Linear(self.lower_size, 1),
                                   nn.Softplus())
            setattr(self, "com_score_fun_{0}".format(i), F_func)

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [s_size, s_size, 1]

        lower_bound = T0 - T1.logsumexp(dim=1)  # torch.log(T1.exp().mean(dim = 1)).mean()

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound.sum()

    def mi_loss(self, encode_out, bert_embed, labels):
        """

        :param encode_out: (bsz, seq_len, embed)
        :param bert_embed: (bsz, seq_len, 768)
        :param bert_embed: (bsz, seq_len, 768)
        :return:
            info nce loss (bsz, seq_len)
        """
        bsz, seq_len, embed = encode_out.shape
        mi_losses = 0

        # for pos_index in range(seq_len):
        #     # (bsz, embed) in pos index
        #     z = encode_out[:, pos_index, :].squeeze()
        #
        #     # embed 紧后 copy bsz 次
        #     ga_index = torch.tensor(range(bsz), device=self.device).unsqueeze(-1).expand(-1, bsz*embed).reshape(-1, embed)
        #     # (bsz*bsz, embed)
        #     z_sum = z.gather(0, ga_index)
        #
        #     for target_index in range(seq_len):
        #         # (bsz, 768) in target index
        #         x = bert_embed[:, target_index, :].squeeze()
        #         # 768 按照顺序 copy bsz 次
        #         x_sum = x.repeat(bsz, 1)
        #
        #         F_func = getattr(self, "com_score_fun_{0}".format(target_index))
        #
        #         # (bsz)
        #         scores = F_func(torch.cat([x, z], dim=-1))
        #         # (bsz*bsz)
        #         scores_sum = F_func(torch.cat([x_sum, z_sum], dim=-1)).reshape(bsz, bsz)
        #
        #         # lower_bound = (scores - scores_sum.logsumexp(dim=0)).sum() + np.log(bsz) * seq_len
        #         # skip constant, minimize (-1) * lower_bound equals maximize lower_bound
        #         lower_bound = (-1) * (scores.squeeze() - scores_sum.logsumexp(dim=-1)).sum()
        #         lower_bounds += lower_bound

        # (bsz, seq_len, embed) in pos index

        # (batch, seq, embed)
        z = encode_out
        z_sum_ori = z.reshape(bsz*seq_len, 1, embed)
        # (bsz*seq_len, bsz, embed)
        z_sum = z_sum_ori.expand(-1, bsz, -1)

        # # embed 紧后 copy bsz 次
        # ga_index = torch.tensor(range(bsz), device=self.device).unsqueeze(-1).expand(-1, bsz*embed).reshape(bsz, -1, embed)
        # # (bsz*bsz, embed)
        # z_sum = z.gather(1, ga_index)

        random_range = int((seq_len-1) * 0.8) + 1
        random_idxes = np.random.randint(0, random_range, int((random_range-1)/3 + 1))
        # mask non entity word
        # (batch, seq)
        token_mask = (labels > 0).int()

        for target_index in random_idxes:
            F_func = getattr(self, "com_score_fun_{0}".format(target_index))

            # (bsz, 1, 768) in target index
            x_ori = bert_embed[:, target_index, :]
            # 按照顺序 copy seq_len 次
            # (batch, seq, 768)
            x = x_ori.unsqueeze(1).expand(-1, seq_len, -1)
            # (batch, seq)
            scores = F_func(torch.cat([x, z], dim=-1)).squeeze()

            # (bsz*seq_len, bsz, 756)
            x_sum = x_ori.reshape(1, bsz, 768).expand(bsz*seq_len, -1, -1)

            # (bsz*seq_len, bsz)
            scores_sum = F_func(torch.cat([x_sum, z_sum], dim=-1)).reshape(bsz, seq_len, bsz)

            # lower_bound = (scores - scores_sum.logsumexp(dim=0)).sum() + np.log(bsz) * seq_len
            # skip constant, minimize (-1) * lower_bound equals maximize lower_bound
            mi_loss = (-1) * (token_mask * (scores.squeeze() - scores_sum.logsumexp(dim=-1))).sum()
            mi_losses += mi_loss

        return mi_losses

    # TODO
    def span_mi_loss(self, x_spans, x_span_idxes, y_spans, y_span_idxes):
        """

        :param x_spans: (bsz, span_num, dim)
        :param x_span_idxes: (bsz)
        :param y_spans: (bsz, span_num, dim)
        :param y_span_idxes: (bsz)
        :return:
        """
        bsz, _, dim = x_spans.shape
        x_span_idxes = x_span_idxes.unsqueeze(1).unsqueeze(1).repeat(1, 1, dim)
        y_span_idxes = y_span_idxes.unsqueeze(1).unsqueeze(1).repeat(1, 1, dim)

        x_spans_con = x_spans.gather(1, x_span_idxes).squeeze()
        y_spans_con = y_spans.gather(1, y_span_idxes).squeeze()

        info_nce_loss = self.forward(x_spans_con, y_spans_con)

        return info_nce_loss * -1



def kl_div(param1, param2):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.

    """
    # u, log(std**2)
    mean1, log_cov1 = param1
    mean2, log_cov2 = param2
    cov1 = log_cov1.exp()
    cov2 = log_cov2.exp()
    bsz, seqlen, tag_dim = mean1.shape
    var_len = tag_dim * seqlen

    cov2_inv = 1 / cov2
    mean_diff = mean1 - mean2

    mean_diff = mean_diff.view(bsz, -1)
    cov1 = cov1.view(bsz, -1)
    cov2 = cov2.view(bsz, -1)
    cov2_inv = cov2_inv.view(bsz, -1)

    temp = (mean_diff * cov2_inv).view(bsz, 1, -1)
    KL = 0.5 * (
            torch.sum(torch.log(cov2), dim=1)
            - torch.sum(torch.log(cov1), dim=1)
            - var_len
            + torch.sum(cov2_inv * cov1, dim=1)
            + torch.bmm(temp, mean_diff.view(bsz, -1, 1)).view(bsz)
    )

    return KL


def kl_norm(mu, log_var):
    """

    :param mu: u
    :param log_var: log(std**2)

    :return:
        D_kl(N(u, std**2), N(0, 1))
    """

    return -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1).mean()
