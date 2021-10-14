from torch import nn
import torch
from .bert_ner import BertPreTrainedModel, BertModel, CRF, valid_sequence_output
from .MI_estimators import CLUB, vCLUB, VIB, kl_norm, InfoNCE
import torch.nn.functional as F


class BertCrfWithBN(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super(BertCrfWithBN, self).__init__(config)
        # just support bn/oov/cc
        assert args.mode in ['bn', 'oov', 'cc']
        self.mode = args.mode

        # ---------------- encoder ------------------
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
                embedding_dim=config.hidden_size,
                hidden_dim=(args.hidden_dim + config.hidden_size) // 2,
                tag_dim=args.hidden_dim,
                device=args.device
            )

        # predict p(y|z), I(Z; Y) 部分（decoder部分）
        # TODO 加入 sample 机制
        self.sample_size = 1
        self.z_classifier = nn.Linear(args.hidden_dim, config.num_labels)
        self.z_decoder = CRF(num_tags=config.num_labels, batch_first=True)

        # predict p(y|v)
        self.alpha = args.alpha     # kl_v_z params
        self.v_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.v_decoder = CRF(num_tags=config.num_labels, batch_first=True)

        # ---------------- OOV regular ------------------
        self.gama = args.gama       # oov regular weights
        self.oov_reg = vCLUB()

        # ---------------- cross category regular ------------------
        self.theta = args.theta     # cross category weights
        self.cross_category_reg = InfoNCE(
            config.hidden_size, config.hidden_size, args.max_seq_length
        )

        self.init_weights()

    def forward(self, decode=False, **kwargs):
        """
        默认有 labels 为 train, 无 labels 为 test
        """
        assert 'origin_features' in kwargs

        if self.mode == 'bn':
            # training (loss, logits), testing (tags)
            outputs = self.information_bottleneck_opt(
                kwargs['origin_features'],
                decode=decode
            )
        elif self.mode == 'oov':
            switched_feas = kwargs.get('switched_features', None)
            # training (loss, logits), testing (tags)
            outputs = self.oov_opt(
                kwargs['origin_features'],
                entity_switched_features=switched_feas,
                decode=decode
            )
        elif self.mode == 'cc':
            pos_feas = kwargs.get('positive_features', None)
            neg_feas = kwargs.get('negative_features', None)
            outputs = self.cross_category_opt(
                kwargs['origin_features'],
                positive_features=pos_feas,
                negative_features=neg_feas,
                decode=decode
            )
        else:
            raise Exception('Do not mode {}'.format(self.mode))

        return outputs  # (loss), scores

    # TODO, add sample mechanism
    def information_bottleneck_opt(self, origin_features, decode=False):
        """
        默认有 labels 为 train, 无 labels 为 test
        """
        v_outputs = self.label_predict(
            **origin_features, v_predict=True
        )

        # p(t|x) 假设符合高斯分布
        attention_mask = origin_features['attention_mask']
        mean, log_var = self.bn_encoder.get_mu_logvar(v_outputs['v'])
        bsz, seqlen, _ = mean.shape

        if 'labels' in origin_features:    # training
            # (bsz * sample_size, seq_len, tag_dim)
            z = mean
            ex_labels = origin_features['labels'].unsqueeze(1)\
                .repeat(1, self.sample_size, 1) \
                .view(bsz * self.sample_size, seqlen)

            attention_mask = origin_features['attention_mask'].unsqueeze(1)\
                .repeat(1, self.sample_size, 1). \
                view(bsz * self.sample_size, seqlen)
        else:   # testing
            z = mean

        logits = self.z_classifier(z)

        if decode:
            tags = self.z_decoder.decode(logits, attention_mask)
            outputs = (tags,)
        else:
            outputs = (logits,)

        if 'labels' in origin_features:
            kl_v_z = self.alpha * F.kl_div(logits.softmax(dim=-1).log(),
                                           v_outputs['logits'].softmax(dim=-1),
                                           reduction='sum')
            loss_dic = {'kl_v_z': kl_v_z}

            # labels 第0个 必须为 'O'
            ex_labels = torch.where(ex_labels >= 0, ex_labels, torch.zeros_like(ex_labels))
            decoder_log_likely = self.z_decoder(
                emissions=logits, tags=ex_labels, mask=attention_mask,
            )
            # first item loss
            loss_dic['z_crf'] = -1 * decoder_log_likely
            loss_dic['v_crf'] = v_outputs['loss']

            # reg p(z|x) to N(0,1)
            if self.beta > 0:
                kl_encoder = kl_norm(mean, log_var)
                loss_dic['norm'] = self.get_beta() * kl_encoder

            loss_dic['loss'] = sum([item[1] for item in loss_dic.items()])
            outputs = (loss_dic,) + outputs

        return outputs  # (loss), scores

    # duplicated code, todo merge different methods
    def cross_category_opt(
            self,
            origin_features,
            positive_features=None,
            negative_features=None,
            decode=False
    ):
        ori_outputs = self.label_predict(
            **origin_features, decode=decode
        )

        if decode:
            outputs = (ori_outputs['tags'],)
        else:
            outputs = (ori_outputs['logits'],)

        if 'labels' in origin_features:
            assert positive_features is not None
            assert negative_features is not None
            loss_dic = {}
            loss_dic['ori'] = ori_outputs['loss']

            pos_outputs = self.label_predict(
                **positive_features, decode=decode
            )
            loss_dic['pos'] = pos_outputs['loss']

            neg_outputs = self.label_predict(
                **negative_features, decode=decode
            )
            loss_dic['neg'] = neg_outputs['loss']

            contrast_loss = self.cross_category_reg.contrast_forward(
                ori_outputs['v'], pos_outputs['v'], neg_outputs['v']
            )
            loss_dic['oov'] = contrast_loss
            loss_dic['loss'] = sum([item[1] for item in loss_dic.items()])
            outputs = (loss_dic,) + outputs

        return outputs  # (loss), scores

    # duplicated code, todo merge different methods
    def oov_opt(
            self,
            origin_features,
            entity_switched_features=None,
            decode=False
    ):
        # 先不做模型蒸馏，直接用 v 做结果预测
        ori_outputs = self.label_predict(
            **origin_features, decode=decode, v_predict=True
        )

        if decode:
            outputs = (ori_outputs['tags'],)
        else:
            outputs = (ori_outputs['logits'],)

        if 'labels' in origin_features:
            assert entity_switched_features is not None
            loss_dic = {'ori': ori_outputs['loss']}

            switched_outputs = self.label_predict(
                **entity_switched_features, decode=decode, v_predict=True
            )

            loss_dic['switch'] = switched_outputs['loss']

            # todo, add I(z1, z2)
            entity_mi = self.oov_reg.update(
                ori_outputs['v'], switched_outputs['v']
            )
            loss_dic['oov'] = self.gama * entity_mi
            loss_dic['loss'] = sum([item[1] for item in loss_dic.items()])
            outputs = (loss_dic,) + outputs

        return outputs  # (loss), scores

    # {loss, v, tags/scores}
    def label_predict(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            valid_mask=None,
            labels=None,
            decode=False,
            v_predict=False
    ):
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
        bsz, seqlen, _ = origin_out.shape

        classifier = self.v_classifier if v_predict else self.z_classifier
        decoder = self.v_decoder if v_predict else self.z_decoder

        logits = classifier(origin_out)
        outputs = {'v': origin_out}

        if decode:
            tags = decoder.decode(logits, attention_mask)
            outputs['tags'] = tags
        else:
            outputs['logits'] = logits

        if labels is not None:
            # labels 第0个 必须为 'O'
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            # attention mask 可以用来 mask 无负例情况下的样本 loss
            decoder_log_likely = decoder(
                emissions=logits, tags=labels, mask=attention_mask
            )
            outputs['loss'] = -1 * decoder_log_likely

        return outputs

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
        # encoder, （layer_num, bsz, seq_len, hidden_size）
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        # 把 padding 部分置零
        # 倒序取值
        # for sequence_output in sequence_outputs[-self.layers_num:][::-1]:
        valid_output, attention_mask = valid_sequence_output(
            sequence_output[0],
            valid_mask,
            attention_mask
        )
        valid_output = self.dropout(valid_output)

        # （layer_num, (bsz, seq_len, hidden_size）)
        return valid_output

    def post_encoding(self, origin_out):
        # p(t|x) 假设符合高斯分布
        means, log_vars = [], []
        for i in range(len(origin_out)):
            bn_encoder = getattr(self, 'poster_encoder_{}'.format(i))
            l_mean, l_log_var = bn_encoder.get_mu_logvar(origin_out[i])
            means.append(l_mean)
            log_vars.append(l_log_var)

        return means, log_vars

    def get_beta(self, step=0):

        # return (step % 500) / 500 * self.beta
        return self.beta
