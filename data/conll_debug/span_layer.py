import torch
from torch import nn
from models.span_extractors import EndpointSpanExtractor
from torch.nn import functional as F


class SpanLayer(torch.nn.Module):
    def __init__(self, hidden_size, token_len_dim, span_len_dim, morph_emb_dim, max_span_len, morph_num):
        #  bucket_widths: Whether to bucket the span widths into log-space
        #  buckets. If `False`, the raw span widths are used.
        #  num_width_embeddings 设为 4 貌似太小, 感觉设置为 7 更合适
        super().__init__()
        self.span_extractor = EndpointSpanExtractor(
            hidden_size,
            num_width_embeddings=4,
            span_width_embedding_dim=token_len_dim,
            bucket_widths=True
        )
        # return dim size: start + end + token len + span len + morph
        # hidden_size * 2 + TOKEN_LEN_DIM + SPAN_LEN_DIM + SPAN_MORPH_DIM

        self.spanLen_embedding = nn.Embedding(
            max_span_len + 1, span_len_dim, padding_idx=0
        )

        self.morph_embedding = nn.Embedding(
            morph_num + 1, morph_emb_dim, padding_idx=0
        )

    def forward(
        self,
        sequences_embed,
        span_idxs_ltoken,
        span_lens,
        morph_idxs
    ):
        # input [batch, seq_len, hidden]
        # output [batch, n_span, hidden]
        all_span_rep = self.span_extractor(sequences_embed, span_idxs_ltoken.long())

        # (bs, n_span, max_spanLen, dim)
        span_morph_rep = self.morph_embedding(morph_idxs)
        span_morph_rep = torch.sum(span_morph_rep, dim=2)  # (bs, n_span, dim)

        spanlen_rep = self.spanLen_embedding(span_lens)  # (bs, n_span, len_dim)
        spanlen_rep = F.relu(spanlen_rep)

        all_span_rep = torch.cat((all_span_rep, spanlen_rep, span_morph_rep), dim=-1)

        return all_span_rep


