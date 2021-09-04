from torch import nn
import torch
from .bert_ner import BertPreTrainedModel, BertModel, CRF, valid_sequence_output

SMALL = 1e-08


class PostEncoder(nn.Module):
    """
    base encoder 设置为 BERT
    """
    def __init__(
        self,
        device,
        embedding_dim=768,
        hidden_dim=500,
        tag_dim=128
    ):
        super(PostEncoder, self).__init__()

        # params
        self.device = device
        self.hidden_dim = hidden_dim
        self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid,
                            'relu': torch.relu}
        self.activation = self.activations['tanh']
        interm_layer_size = (embedding_dim + hidden_dim) // 2

        # additional encoder
        self.linear_layer = nn.Linear(embedding_dim, interm_layer_size)
        self.linear_layer3 = nn.Linear(interm_layer_size, hidden_dim)

        # ============= Covariance matrix & Mean vector ================
        self.hidden2mean = nn.Linear(hidden_dim, tag_dim)
        self.hidden2std = nn.Linear(hidden_dim, tag_dim)

    def forward_sent_batch(self, embeds):

        temps = self.activation(self.linear_layer(embeds))
        temps = self.activation(self.linear_layer3(temps))
        mean = self.hidden2mean(temps)  # bsz, seqlen, dim
        std = self.hidden2std(temps)  # bsz, seqlen, dim
        cov = std * std + SMALL
        return mean, cov

    def get_sample_from_param_batch(self, mean, cov, sample_size):
        bsz, seqlen, tag_dim = mean.shape
        z = torch.randn(bsz, sample_size, seqlen, tag_dim).to(self.device)

        z = z * torch.sqrt(cov).unsqueeze(1).expand(-1, sample_size, -1, -1) + \
            mean.unsqueeze(1).expand(-1, sample_size, -1, -1)

        return z.view(-1, seqlen, tag_dim)

    def get_statistics_batch(self, elmo_embeds):
        mean, cov = self.forward_sent_batch(elmo_embeds)
        return mean, cov


class BertCrfWithBN(BertPreTrainedModel):
    def __init__(
        self,
        config,
        args=None
    ):
        super(BertCrfWithBN, self).__init__(config)
        # encoder 部分
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.post_encoder = PostEncoder(
            args.device,
            embedding_dim=config.hidden_size,
            hidden_dim=(args.hidden_dim+config.hidden_size)//2,
            tag_dim=args.hidden_dim
        )

        self.sample_size = args.sample_size
        self.regular_z = args.regular_z

        # entity regular_z
        self.gama = args.get("gama", 1e-5)
        self.regular_entity = args.get("regular_entity", False)

        # decoder 部分
        self.classifier = nn.Linear(args.hidden_dim, config.num_labels)
        self.decoder = CRF(num_tags=config.num_labels, batch_first=True)
        # self.init_weights()

        # r(t)
        self.beta = args.beta
        self.r_mean = nn.Parameter(torch.randn(args.max_seq_length, args.hidden_dim))
        self.r_std = nn.Parameter(torch.randn(args.max_seq_length, args.hidden_dim))

        self.init_weights()

    def kl_div(self, param1, param2):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        mean1, cov1 = param1
        mean2, cov2 = param2
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

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        valid_mask=None,
        labels=None,
        decode=False
    ):
        """
        默认有 labels 为 train, 无 labels 为 test
        """
        # encoder
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        # 把 padding 部分置零
        sequence_output, attention_mask = valid_sequence_output(
            sequence_output,
            valid_mask,
            attention_mask
        )

        sequence_output = self.dropout(sequence_output)

        # p(t|x) 基于采样得到的 mean 和 cov (假设符合高斯分布)
        mean, cov = self.post_encoder.get_statistics_batch(sequence_output)
        bsz, seqlen, _ = sequence_output.shape

        # train sample by IID, test by argmax
        # (bsz * sample_size, seq_len, tag_dim)
        if labels is not None:
            t = self.post_encoder.get_sample_from_param_batch(
                mean, cov, self.sample_size
            )
            # labels expand
            labels = labels.unsqueeze(1).repeat(1, self.sample_size, 1) \
                .view(bsz * self.sample_size, seqlen)

            attention_mask = attention_mask.unsqueeze(1). \
                repeat(1, self.sample_size, 1). \
                view(bsz * self.sample_size, seqlen)
        else:
            t = mean

        logits = self.classifier(t)

        if decode:
            tags = self.decoder.decode(logits, attention_mask)
            outputs = (tags,)
        else:
            outputs = (logits,)

        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            # reduction 是否为 mean 待确认
            decoder_log_likely = self.decoder(
                emissions=logits, tags=labels, mask=attention_mask,
                # reduction="mean"
            )
            # first item loss
            nlpy_t = -1 * decoder_log_likely

            if self.regular_z:
                mean_r = self.r_mean[:seqlen].unsqueeze(0).expand(bsz, -1, -1)
                std_r = self.r_std[:seqlen].unsqueeze(0).expand(bsz, -1, -1)
                cov_r = std_r * std_r + SMALL

                # second item loss
                kl_div = self.kl_div((mean, cov), (mean_r, cov_r))
                nlpy_t += self.beta * kl_div.mean()

            outputs = (nlpy_t,) + outputs

        return outputs  # (loss), scores
