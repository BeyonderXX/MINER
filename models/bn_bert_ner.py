from torch import nn
import torch
from .bert_ner import BertPreTrainedModel, BertModel, CRF, valid_sequence_output
from .MI_estimators import CLUB, vCLUB, VIB, kl_div, kl_norm, InfoNCE
from .span_extractors import EndpointSpanExtractor
from .classifier import MultiNonLinearClassifier
from torch.nn import functional as F


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

        # default just use last layer out
        self.layers_num = 1

        self.post_encoder = []
        tag_dim = int(args.hidden_dim/self.layers_num)
        hidden_dim = tag_dim * self.layers_num

        self.post_encoder = MI_estimator(
                embedding_dim=config.hidden_size,
                hidden_dim=(tag_dim + config.hidden_size) // 2,
                tag_dim=tag_dim,
                device=args.device
            )
        # # poster encoder 编号与bert layer顺序相反
        # setattr(self, 'poster_encoder_{}'.format(i), post_encoder)

        # r(t)
        self.r_mean = nn.Parameter(torch.randn(args.max_seq_length, hidden_dim))
        self.r_log_var = nn.Parameter(torch.randn(args.max_seq_length, hidden_dim))

        self.entity_regularizer = vCLUB()

        # I(Z; Y) 部分（decoder部分）
        self.sample_size = args.sample_size

        if self.baseline:
            # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.classifier = torch.nn.Softmax(dim=-1)
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



        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)

        # self.span_embedding = SingleLinearClassifier(config.hidden_size * 2, 1)

        self.hidden_size = config.hidden_size

        self.span_combination_mode = 'x,y'
        self.max_span_width = 4
        self.n_class = 5
        self.tokenLen_emb_dim = 50 # must set, when set a value to the max_span_width.


        #  bucket_widths: Whether to bucket the span widths into log-space buckets. If `False`, the raw span widths are used.

        self._endpoint_span_extractor = EndpointSpanExtractor(config.hidden_size,
                                                              combination=self.span_combination_mode,
                                                              num_width_embeddings=self.max_span_width,
                                                              span_width_embedding_dim=self.tokenLen_emb_dim,
                                                              bucket_widths=True)


        self.linear = nn.Linear(10, 1)
        self.score_func = nn.Softmax(dim=-1)

        # import span-length embedding
        self.spanLen_emb_dim =100
        self.morph_emb_dim = 100

        input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim + self.spanLen_emb_dim + self.morph_emb_dim


        self.span_embedding = MultiNonLinearClassifier(input_dim, self.n_class, dropout_rate=0.2)

        self.spanLen_embedding = nn.Embedding(4+1, self.spanLen_emb_dim, padding_idx=0)

        self.morph_embedding = nn.Embedding(5+1, self.morph_emb_dim, padding_idx=0)

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
        step=0,
        # separate
        span_idxs_ltoken = None,
        morph_idxs = None,
        span_label_ltoken = None,
        span_lens = None,
        span_weights = None,
        real_span_mask_ltoken = None,
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
        # origin_out = self.encoding(
        origin_out, layer_out = self.encoding(
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

        t = origin_out

        predicts = self.classifier(t) # self.classifier = torch.nn.Softmax(dim=-1)
        
        outputs = {}

        if decode:
            outputs['pred'] = predicts

        if labels is not None:
            loss = self.compute_loss(t, span_label_ltoken, real_span_mask_ltoken, span_weights, mode='train')
            
            outputs[f"train_loss"] = loss
            outputs['loss'] = loss


        return outputs


    def compute_loss(self, all_span_rep, span_label_ltoken, real_span_mask_ltoken, span_weights, mode):
        '''

        :param all_span_rep: shape: (bs, n_span, n_class)
        :param span_label_ltoken:
        :param real_span_mask_ltoken:
        :return: loss
        '''
        batch_size, n_span = span_label_ltoken.size()
        # print('batch_size = :')
        # print(batch_size)
        # print(n_span)

        all_span_rep1 = all_span_rep.view(-1,self.n_class)
        span_label_ltoken1 = span_label_ltoken.view(-1)

        loss = self.cross_entropy(all_span_rep1, span_label_ltoken1)
        loss = loss.view(batch_size, n_span)
        # print('loss 1: ', loss)
        if mode=='train':
            span_weight = span_weights
            loss = loss*span_weight
            # print('loss 2: ', loss)

        loss = torch.masked_select(loss, real_span_mask_ltoken.bool())

        # print("1 loss: ", loss)
        loss= torch.mean(loss)
        # print("loss: ", loss)

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
        # encoder
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0] # [batch, seq_len, hidden]
        # print('\n\nsequence_output_size:')
        # print(sequence_output.size())
        all_span_rep = self._endpoint_span_extractor(sequence_output, span_idxs_ltoken.long()) # [batch, n_span, hidden]

        # print('\nall_span_rep_size:')
        # print(all_span_rep.size())        
        
        
        span_morph_rep = self.morph_embedding(morph_idxs) #(bs, n_span, max_spanLen, dim)
        # print('\nspan_morph_rep_size:')
        # print(span_morph_rep.size()) 
        span_morph_rep = torch.sum(span_morph_rep, dim=2) #(bs, n_span, dim)
        # print('\nspan_morph_rep_size:')
        # print(span_morph_rep.size()) 
        spanlen_rep = self.spanLen_embedding(span_lens)  # (bs, n_span, len_dim)
        spanlen_rep = F.relu(spanlen_rep)
        # print('\nspanlen_rep_size:')
        # print(spanlen_rep.size()) 
        all_span_rep = torch.cat((all_span_rep, spanlen_rep, span_morph_rep), dim=-1)
        # print('\nall_span_rep_size:')
        # print(all_span_rep.size()) 
        all_span_rep = self.span_embedding(all_span_rep)  # (batch, n_span, n_class)
        # print('\nall_span_rep_size:')
        # print(all_span_rep.size()) 

        return all_span_rep, outputs[2]


        # valid_output, attention_mask = valid_sequence_output(
        #     sequence_output,
        #     valid_mask,
        #     attention_mask
        # )
        # valid_output = self.dropout(valid_output)

        # （layer_num, (bsz, seq_len, hidden_size）)
        return valid_output, outputs[2]
        # return valid_output



    




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
