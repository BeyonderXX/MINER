from torch import nn
import torch
from .bert_ner import BertPreTrainedModel, BertModel, CRF, valid_sequence_output
from .MI_estimators import CLUB, vCLUB, VIB, kl_div, kl_norm, InfoNCE
from .span_extractors import EndpointSpanExtractor
from .classifier import MultiNonLinearClassifier
from torch.nn import functional as F


class BertSpanNerBN(BertPreTrainedModel):
    def __init__(
        self,
        config,
        args=None
    ):
        super(BertSpanNerBN, self).__init__(config)
        # just support bn/oov/cc
        assert args.mode in ['bn', 'oov', 'cc']
        self.mode = args.mode

        # ---------------- encoder ------------------
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)

        self.span_combination_mode = 'x,y'
        self.max_span_width = 4
        self.n_class = 5
        # must set, when set a value to the max_span_width.
        self.tokenLen_emb_dim = 50

        #  bucket_widths: Whether to bucket the span widths into log-space
        #  buckets. If `False`, the raw span widths are used.
        self.span_extractor = EndpointSpanExtractor(
            config.hidden_size,
            combination=self.span_combination_mode,
            num_width_embeddings=self.max_span_width,
            span_width_embedding_dim=self.tokenLen_emb_dim,
            bucket_widths=True
        )

        # import span-length embedding
        self.spanLen_emb_dim = 100
        self.morph_emb_dim = 100

        # start + end + token len + span len + morph
        input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim \
                    + self.spanLen_emb_dim + self.morph_emb_dim

        # class logits
        self.span_classifier = MultiNonLinearClassifier(
            args.hidden_dim, self.n_class, dropout_rate=0.2
        )
        self.softmax = torch.nn.Softmax(dim=-1)

        self.spanLen_embedding = nn.Embedding(
            4+1, self.spanLen_emb_dim, padding_idx=0
        )

        self.morph_embedding = nn.Embedding(
            5+1, self.morph_emb_dim, padding_idx=0
        )

        # ---------------- info bottleneck ------------------
        if args.mi_estimator == 'VIB':
            MI_estimator = VIB
        elif args.mi_estimator == 'CLUB':
            # TODO, support
            MI_estimator = CLUB
        else:
            raise ValueError('Do not support {} estimator!'.format(args.mi_estimator))

        self.beta = args.beta   # norm reg weights
        self.bn_encoder = MI_estimator(
                embedding_dim=input_dim,
                hidden_dim=(args.hidden_dim + input_dim) // 2,
                tag_dim=args.hidden_dim,
                device=args.device
            )

        # predict p(y|z), I(Z; Y) 部分（decoder部分）
        # TODO 加入 sample 机制
        self.sample_size = 1

        # self.z_classifier = nn.Linear(args.hidden_dim, config.num_labels)

        # predict p(y|v)
        # self.alpha = args.alpha     # kl_v_z params

        # ---------------- OOV regular ------------------
        self.gama = args.gama       # oov regular weights
        self.oov_reg = vCLUB()

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
        decode=False,
        step=0,
        # separate
        span_idxs_ltoken=None,
        morph_idxs=None,
        span_label_ltoken=None,
        span_lens=None,
        span_weights=None,
        real_span_mask_ltoken=None,

        trans_input_ids=None,
        trans_attention_mask=None,
        trans_valid_mask=None,
        trans_token_type_ids=None,
        # separate = None,
        trans_span_idxs_ltoken = None,
        trans_morph_idxs = None,
        trans_span_label_ltoken = None,
        trans_span_lens = None,
        trans_span_weights = None,
        trans_real_span_mask_ltoken = None
    ):
        """
        默认有 labels 为 train, 无 labels 为 test
        """
        origin_rep = self.encoding(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            valid_mask,
            span_idxs_ltoken,
            span_lens,
            morph_idxs
        )
        mean, log_var = self.bn_encoder.get_mu_logvar(origin_rep)
        bsz, seqlen, _ = mean.shape

        if labels is not None:  # training
            # (bsz * sample_size, seq_len, tag_dim)
            z = mean
            # TODO, add span_label_ltoken
            ex_span_label_ltoken = span_label_ltoken.unsqueeze(1) \
                .repeat(1, self.sample_size, 1) \
                .view(bsz * self.sample_size, seqlen)
            ex_real_span_mask_ltoken = real_span_mask_ltoken.unsqueeze(1) \
                .repeat(1, self.sample_size, 1) \
                .view(bsz * self.sample_size, seqlen)
            ex_span_weights = span_weights.unsqueeze(1) \
                .repeat(1, self.sample_size, 1) \
                .view(bsz * self.sample_size, seqlen)
            
        else:  # testing
            z = mean

        # (batch, n_span, n_class)
        logits = self.span_classifier(z)
        outputs = []
        loss_dict = {}

        if decode:
            predicts = self.softmax(logits)
            outputs = [predicts]

        if labels is not None:
            loss = self.compute_loss(
                logits, ex_span_label_ltoken, ex_real_span_mask_ltoken, 
                ex_span_weights, mode='train'
            )
            loss_dict['z_loss'] = loss

            # reg p(z|x) to N(0,1)
            if self.beta > 0:
                kl_encoder = kl_norm(mean, log_var)
                loss_dict['norm'] = self.get_beta() * kl_encoder

            # add switch features
            switch_rep = self.encoding(
                trans_input_ids,
                trans_attention_mask,
                trans_token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                trans_valid_mask,
                trans_span_idxs_ltoken,
                trans_span_lens,
                trans_morph_idxs
            )
            switch_mean, switch_log_var = self.bn_encoder.get_mu_logvar(switch_rep)
            switch_logits = self.span_classifier(switch_mean)

            switch_loss = self.compute_loss(
                switch_logits, trans_span_label_ltoken,
                trans_real_span_mask_ltoken, trans_span_weights, mode='train'
            )
            loss_dict['s_z_loss'] = switch_loss

            # todo, add I(z1, z2)
            entity_mi = self.oov_reg.update(
                mean, switch_mean
            )
            loss_dict['oov'] = self.gama * entity_mi

            loss_dict['loss'] = sum([item[1] for item in loss_dict.items()])
            outputs = [loss_dict] + outputs

        return outputs  # (loss), scores

    def compute_loss(self, all_span_rep, span_label_ltoken, real_span_mask_ltoken, span_weights, mode):
        """

        :param all_span_rep: shape: (bs, n_span, n_class)
        :param span_label_ltoken:
        :param real_span_mask_ltoken:
        :return: loss
        """
        batch_size, n_span = span_label_ltoken.size()

        all_span_rep1 = all_span_rep.view(-1, self.n_class)
        span_label_ltoken1 = span_label_ltoken.view(-1)

        loss = self.cross_entropy(all_span_rep1, span_label_ltoken1)
        loss = loss.view(batch_size, n_span)

        if mode == 'train':
            span_weight = span_weights
            loss = loss*span_weight

        loss = torch.masked_select(loss, real_span_mask_ltoken.bool())
        loss = torch.mean(loss)

        return loss

    def encoding(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        valid_mask, 
        span_idxs_ltoken,
        span_lens,
        morph_idxs
    ):
        # encoder [batch, seq_len, hidden]
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        # try:
        # [batch, n_span, hidden]
        all_span_rep = self.span_extractor(sequence_output[0], span_idxs_ltoken.long())
        # except:
        #     print('debug')
        # (bs, n_span, max_spanLen, dim)
        span_morph_rep = self.morph_embedding(morph_idxs)

        # TODO，sum 会导致模型无法分辨 span 特征来自哪个token，直接 reshape 效果待测试
        span_morph_rep = torch.sum(span_morph_rep, dim=2)  # (bs, n_span, dim)

        spanlen_rep = self.spanLen_embedding(span_lens)  # (bs, n_span, len_dim)
        spanlen_rep = F.relu(spanlen_rep)
        all_span_rep = torch.cat((all_span_rep, spanlen_rep, span_morph_rep), dim=-1)

        return all_span_rep

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
