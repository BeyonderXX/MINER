from torch import nn
import torch
from .bert_ner import BertPreTrainedModel, BertModel, CRF, valid_sequence_output
from .MI_estimators import CLUB, vCLUB, VIB, kl_div, kl_norm


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

        # whether use baseline model
        self.baseline = args.baseline

        # post encoder 部分
        self.beta = args.beta
        self.regular_z = args.regular_z
        self.reg_norm = args.regular_norm

        if args.mi_estimator is 'VIB':
            MI_estimator = VIB
        elif args.mi_estimator is 'CLUB':
            # TODO, support
            MI_estimator = CLUB
        else:
            raise ValueError('Do not support {} estimator!'.format(args.mi_estimator))

        self.post_encoder = MI_estimator(
            embedding_dim=config.hidden_size,
            hidden_dim=(args.hidden_dim + config.hidden_size) // 2,
            tag_dim=args.hidden_dim,
            device=args.device
        )
        # r(t)
        self.r_mean = nn.Parameter(torch.randn(args.max_seq_length, args.hidden_dim))
        self.r_log_var = nn.Parameter(torch.randn(args.max_seq_length, args.hidden_dim))

        # I(Z; Y) 部分（decoder部分）
        self.sample_size = args.sample_size

        if self.baseline:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = nn.Linear(args.hidden_dim, config.num_labels)
        self.decoder = CRF(num_tags=config.num_labels, batch_first=True)

        # I(Z_i;X_i | X_(j!=i))
        self.regular_entity = args.regular_entity
        self.gama = args.gama
        self.entity_regularizer = vCLUB()

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        valid_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        # separate
        trans_input_ids=None,
        trans_attention_mask=None,
        trans_valid_mask=None,
        trans_token_type_ids=None,
        trans_labels=None,
        trans_position_ids=None,
        trans_head_mask=None,
        trans_inputs_embeds=None,
        decode=False,
        step=0
    ):
        """
        默认有 labels 为 train, 无 labels 为 test
        """
        origin_out = self.encoding(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            valid_mask
        )

        # p(t|x) 假设符合高斯分布

        mean, log_var = self.post_encoder.get_mu_logvar(origin_out)
        bsz, seqlen, _ = mean.shape

        # train sample by IID, test by argmax
        if self.baseline:
            t = origin_out
        elif labels is not None and self.sample_size > 0:
            # (bsz * sample_size, seq_len, tag_dim)
            t = self.post_encoder.get_sample_from_param_batch(
                mean, log_var, self.sample_size
            )
            # labels expand, (bsz * sample_size, seq_len)
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
            # labels 第0个 必须为 'O'
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            # reduction 是否为 mean 待确认
            decoder_log_likely = self.decoder(
                emissions=logits, tags=labels, mask=attention_mask,
                # reduction="mean"
            )
            # first item loss
            nlpy_t = -1 * decoder_log_likely

            # reg p(z|x) to N(0,1)
            if self.reg_norm:
                kl_encoder = kl_norm(mean, log_var)
                nlpy_t = nlpy_t + self.get_beta(step) * kl_encoder

            if self.regular_z:
                mean_r = self.r_mean[:seqlen].unsqueeze(0).expand(bsz, -1, -1)
                log_var_r = self.r_log_var[:seqlen].unsqueeze(0).expand(bsz, -1, -1)

                # second item loss
                kl = kl_div((mean, log_var), (mean_r, log_var_r))
                nlpy_t += self.beta * kl.mean()

            if self.regular_entity:
                trans_out = self.encoding(
                    trans_input_ids,
                    trans_attention_mask,
                    trans_token_type_ids,
                    trans_position_ids,
                    trans_head_mask,
                    inputs_embeds,
                    trans_valid_mask
                )
                # upper bound of entity
                entity_mi = self.entity_regularizer.update(origin_out,
                                                           trans_out)
                nlpy_t += self.gama * entity_mi

            outputs = (nlpy_t,) + outputs

        return outputs  # (loss), scores

    def encoding(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        valid_mask
    ):
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

        return sequence_output

    def post_encoding(self, sequence_output):
        # p(t|x) 基于采样得到的 mean 和 cov (假设符合高斯分布)
        mean, cov = self.post_encoder.get_statistics_batch(sequence_output)

        return mean, cov

    def get_beta(self, step=0):

        # return (step % 500) / 500 * self.beta
        return self.beta
