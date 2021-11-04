import torch
import random


def valid_sequence_output(sequence_output, valid_mask, attention_mask):
    """
    如果 word 被 tokenizer 切分， 仅保留第一个词的预测token和其原始label对应

    :param sequence_output:
    :param valid_mask:
    :param attention_mask:
    :return:
    """
    batch_size, max_len, feat_dim = sequence_output.shape
    valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32,
                               device='cuda' if torch.cuda.is_available() else 'cpu')
    valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long,
                                       device='cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(batch_size):
        jj = -1
        for j in range(max_len):
            if valid_mask[i][j].item() == 1:
                jj += 1
                valid_output[i][jj] = sequence_output[i][j]
                valid_attention_mask[i][jj] = attention_mask[i][j]
    return valid_output, valid_attention_mask


# 随机选取 实体span
def get_random_span(x_span, x_weights, y_span, y_weights,
                    span_weight=1.0, O_ratio=0.1):
    assert x_span.shape[:-1] == x_weights.shape
    assert y_span.shape[:-1] == y_weights.shape
    bsz, x_span_num, _ = x_span.shape
    _, y_span_num, _ = y_span.shape

    x_span_idxes = torch.zeros(
        bsz, dtype=torch.int64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    y_span_idxes = torch.zeros(
        bsz, dtype=torch.int64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    for i in range(bsz):
        span_idx = [[], []]

        for j in range(x_span_num):
            if x_weights[i][j] >= span_weight:
                span_idx[0].append(j)
        for j in range(y_span_num):
            if y_weights[i][j] >= span_weight:
                span_idx[1].append(j)

        assert len(span_idx[0]) == len(span_idx[1])

        # without entity situation
        if len(span_idx[0]) == 0:
            span_idx[0] = 0
            span_idx[1] = 0
        else:
            #span_idx[0] = random.choice(span_idx[0])
            #span_idx[1] = random.choice(span_idx[1])
            index = random.choice(range(len(span_idx[0])))
            span_idx[0] = span_idx[0][index]
            span_idx[1] = span_idx[1][index]

        x_span_idxes[i] = span_idx[0]
        y_span_idxes[i] = span_idx[1]

    return x_span_idxes, y_span_idxes


def span_select(x_spans, x_span_idxes, y_spans, y_span_idxes):
    """

    :param x_spans: (bsz, span_num, dim)
    :param x_span_idxes: (bsz)
    :param y_spans: (bsz, span_num, dim)
    :param y_span_idxes: (bsz)
    :return:
    """
    bsz, sample_size, _, dim = x_spans.shape
    x_span_idxes = x_span_idxes.unsqueeze(1).unsqueeze(1).repeat(1, 1, dim)
    y_span_idxes = y_span_idxes.unsqueeze(1).unsqueeze(1).repeat(1, 1, dim)

    try:
        x_spans_con = x_spans[:, 0, :, :].squeeze(axis=1).gather(1, x_span_idxes).squeeze(1)
        y_spans_con = y_spans[:, 0, :, :].squeeze(axis=1).gather(1, y_span_idxes).squeeze(1)

        return x_spans_con, y_spans_con
    except Exception:
        print('Error!')

