from torch import nn
import torch
from .bert_ner import BertPreTrainedModel, BertModel, CRF, valid_sequence_output
from .MI_estimators import CLUB, vCLUB, VIB, kl_div, kl_norm, InfoNCE


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

        self.layers_num = 1 if not args.multi_layers else self.bert.config.num_hidden_layers + 1
        assert self.layers_num == 1, "Slow mode for layers num {}, forbidden!".format(self.layers_num)

        self.post_encoder = []
        tag_dim = int(args.hidden_dim/self.layers_num)
        hidden_dim = tag_dim * self.layers_num

        for i in range(self.layers_num):
            post_encoder = MI_estimator(
                    embedding_dim=config.hidden_size,
                    hidden_dim=(tag_dim + config.hidden_size) // 2,
                    tag_dim=tag_dim,
                    device=args.device
                )
            # poster encoder 编号与bert layer顺序相反
            setattr(self, 'poster_encoder_{}'.format(i), post_encoder)

        # r(t)
        self.r_mean = nn.Parameter(torch.randn(args.max_seq_length, hidden_dim))
        self.r_log_var = nn.Parameter(torch.randn(args.max_seq_length, hidden_dim))

        self.entity_regularizer = vCLUB()

        # I(Z; Y) 部分（decoder部分）
        self.sample_size = args.sample_size

        if self.baseline:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = nn.Linear(hidden_dim, config.num_labels)
        self.decoder = CRF(num_tags=config.num_labels, batch_first=True)

        # I(Z_i;X_i | X_(j!=i))
        self.regular_entity = args.regular_entity
        self.gama = args.gama

        self.regular_context = args.regular_context
        self.theta = args.theta
        self.context_regularizer = InfoNCE(config.hidden_size, args.hidden_dim,
                                           args.max_seq_length, device=args.device)

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
        decode=False,
        step=0
    ):
        """
        默认有 labels 为 train, 无 labels 为 test
        """
        origin_out, layer_out = self.encoding(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            valid_mask
        )

        # p(t|x) 假设符合高斯分布
        means, log_vars = self.post_encoding(origin_out)
        mean = torch.cat(means, -1)
        log_var = torch.cat(log_vars, -1)
        bsz, seqlen, _ = means[0].shape

        # train sample by IID, test by argmax
        if self.baseline:
            t = torch.cat(origin_out, -1)
            # context experiment
            context_embedding = self.get_context_embedding(layer_out[0])
            t = t + context_embedding
        elif labels is not None and self.sample_size > 0:
            # (bsz * sample_size, seq_len, tag_dim)
            post_encoder = getattr(self, 'poster_encoder_{}'.format(0))
            t = post_encoder.get_sample_from_param_batch(
                        mean, log_var, self.sample_size
                    )

            # labels expand, (bsz * sample_size, seq_len)
            ex_labels = labels.unsqueeze(1).repeat(1, self.sample_size, 1) \
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
            ex_labels = torch.where(ex_labels >= 0, ex_labels, torch.zeros_like(ex_labels))
            # reduction 是否为 mean 待确认
            decoder_log_likely = self.decoder(
                emissions=logits, tags=ex_labels, mask=attention_mask,
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
                trans_out, trans_layer_out = self.encoding(
                    trans_input_ids,
                    trans_attention_mask,
                    trans_token_type_ids,
                    position_ids,
                    head_mask,
                    inputs_embeds,
                    trans_valid_mask
                )
                # upper bound of entity
                # 如果 bert fix 了， 这里对bert输出 minimize 因为参数量过小，效果很差
                trans_means, log_vars = self.post_encoding(trans_out)
                entity_mi = self.entity_regularizer.update(
                    mean, torch.cat(trans_means, -1)
                )
                nlpy_t += self.gama * entity_mi

            if self.regular_context:
                context_mi_loss = self.context_regularizer.mi_loss(
                    mean, layer_out[0], labels
                )
                nlpy_t += self.theta * context_mi_loss

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
        # （layer_num, bsz, seq_len, hidden_size）
        sequence_outputs = outputs[2]

        # 把 padding 部分置零
        valid_outputs = []

        # 倒序取值
        for sequence_output in sequence_outputs[-self.layers_num:][::-1]:
            valid_output, attention_mask = valid_sequence_output(
                sequence_output,
                valid_mask,
                attention_mask
            )
            valid_output = self.dropout(valid_output)
            valid_outputs.append(valid_output)

        # （layer_num, (bsz, seq_len, hidden_size）)
        return valid_outputs, sequence_outputs

    # 用来测试 context 对 bsl 的提升
    def get_context_embedding(self, base_embedding):
        # (bsz, seq_len, embedding_size)
        bsz, seq_len, embedding_size = base_embedding.shape
        position_masks = torch.eye(seq_len, device=self.device)

        # (seq_len, seq_len)
        position_mask = (position_masks[range(seq_len)] < 1).int() * (1 / seq_len)
        # (bsz * embedding_zie, seq_len)
        base_embedding = base_embedding.view(bsz, embedding_size, seq_len).view(bsz * embedding_size, seq_len)
        # (bsz * embedding_zie, seq_len)
        mask_embedding = torch.mm(base_embedding, position_mask)
        mask_embedding = mask_embedding.view(bsz, embedding_size, seq_len).view(bsz, seq_len, embedding_size)

        return mask_embedding

    def post_encoding(self, origin_out):
        # p(t|x) 假设符合高斯分布
        means, log_vars = [], []
        for i in range(len(origin_out)):
            post_encoder = getattr(self, 'poster_encoder_{}'.format(i))
            l_mean, l_log_var = post_encoder.get_mu_logvar(origin_out[i])
            means.append(l_mean)
            log_vars.append(l_log_var)

        return means, log_vars

    def get_beta(self, step=0):

        # return (step % 500) / 500 * self.beta
        return self.beta
