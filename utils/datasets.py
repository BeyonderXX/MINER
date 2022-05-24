import logging
import os
import random
import torch
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from .allen_utils import enumerate_spans
from .utils_contrastive import InputExample, build_contrast_examples

logger = logging.getLogger(__name__)
sequence_a_segment_id = 0
# define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
cls_token_segment_id = 0
pad_token_label_id = CrossEntropyLoss().ignore_index
morph2idx = {'isupper': 1, 'islower': 2, 'istitle': 3, 'isdigit': 4, 'other': 5}

# examples = load_examples(data_dir, mode, tokenizer)


class SpanNerDataset(Dataset):
    def __init__(self, examples, args=None, tokenizer=None, labels=None, dev=False):
        self.examples = examples
        self.cont_ent = None
        self.cont_ent_idxes = self.get_rand_ent_idxes(examples, max_seq_len=args.max_seq_len)
        self.nume_feas = self.convert_features(examples, tokenizer, labels, max_seq_len=args.max_seq_len,
                                               max_span_len=args.max_span_len, sep_token_extra=False)

        self.cont_examples = self.examples if dev else \
            build_contrast_examples(examples, tokenizer, self.cont_ent_idxes, pmi_json=args.pmi_json,
                                    entity_json=args.entity_json, max_seq_len=args.max_seq_len)

        self.cont_nume_feas = self.nume_feas if dev else \
            self.convert_features(self.cont_examples, tokenizer, labels, max_seq_len=args.max_seq_len,
                                  max_span_len=args.max_span_len, sep_token_extra=False)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.nume_feas[idx], self.cont_nume_feas[idx]

    def get_rand_ent_idxes(self, examples, max_seq_len=128):
        """
        长度为 n 的句子，最大 span 长度为 w，如：

        长度为 6，最长 span 长度为 3，其遍历 span 为：
        I like eat chinese food .
        [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [2, 4],
         [3, 3], [3, 4], [3, 5], [4, 4], [4, 5], [5, 5]]

        长度为 5，最长 span 长度为 4，其遍历 span 为：
        I like eat chinese food
        [[0, 0], [0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3], [1, 4], [2, 2],
        [2, 3], [2, 4], [3, 3], [3, 4], [4, 4]]

        span 个数为 n * w - (w - 1) * w / 2
        给定 span 序列起始索引 [a, b], 其对应的序号为：
        1. b > n - 1 越界， span 不合法；
        2. b <= n - 1,  其序号为 a * w + (b - a)

        """
        cont_ent_idxes = []

        for example in examples:
            # 仍然有可能越界， 这里是word的起始位置，max_seq_len是token索引
            ent_span_idxes = [entity[1:] for entity in example.entities if entity[2] < max_seq_len]
            if ent_span_idxes:
                cont_ent_idxes.append(random.choice(range(len(ent_span_idxes))))
            else:   # no entity instance
                cont_ent_idxes.append(0)

        return cont_ent_idxes

    def convert_features(self, examples, tokenizer, labels,
                         max_seq_len=128, max_span_len=4,
                         ent_weight=1.0, non_ent_weight=0.5,
                         pad_token_segment_id=0, pad_label_id=0,
                         sep_token_extra=False):
        """
        将 example 转为数值特征。

        Return:
            input_ids: token id (after padding)
            input_mask: token mask，which is required by BERT
            segment_ids: BERT sentence segment id
            span_token_idxes: list of (span_start_token_index, span_end_token_index)
            span_labels: span label indexes
            span_weights: span loss weight, 0.5 for None, 1 for entity span
            morph_idxes: morph feature of span
            span_lens: length feature of span
            span_masks: span mask, which is required by SpanNER
            cont_span_idx: index of contrastive ent span
            span_word_idxes: list of (span_start_word_index, span_end_word_index)
        """
        numeric_features = []
        label_map = {ent_type: idx for idx, ent_type in enumerate(labels)}

        for idx, example in enumerate(examples):
            tokens, valid_mask = example.tokens, example.valid_mask
            # all span (sidx, eidx)
            ent_span_idxes = [entity[1:] for entity in example.entities]
            span_word_idxes = enumerate_spans(example.words, offset=0, max_span_width=max_span_len)
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            max_token_num = max_seq_len - special_tokens_count

            span_token_idxes, n_keep = convert_2_token_idx(example.offsets, span_word_idxes, max_len=max_token_num)
            # assert len(span_token_idxes) == n_keep

            # begin{compute the span weight}
            span_weights = []  # return
            span_labels = []
            cont_span_idx = 0

            for i, span_idx in enumerate(span_word_idxes[: n_keep]):
                if span_idx in ent_span_idxes:
                    span_weights.append(ent_weight)
                    span_type = example.entities[ent_span_idxes.index(span_idx)][0]
                    span_labels.append(label_map[span_type])
                    # 正例和负例的对比实体，在各自实体列表里位置相同（前提负例目标实体不越界）
                    if span_idx == ent_span_idxes[self.cont_ent_idxes[idx]]:
                        cont_span_idx = i
                else:
                    span_weights.append(non_ent_weight)
                    span_labels.append(0)

            span_lens = [idxes[1] - idxes[0] + 1 for idxes in span_word_idxes[: n_keep]]  # return
            morph_idxes = get_case_feature(span_word_idxes[: n_keep], example.words, max_span_len)
            span_masks = np.ones_like(span_labels).tolist()

            tmp_minus = int((max_span_len - 1) * max_span_len / 2)
            max_num_span = 128 * max_span_len - tmp_minus

            # prepare span relate inputs
            span_masks = span_padding(span_masks, value=0, max_len=max_num_span)
            span_labels = span_padding(span_labels, value=pad_label_id, max_len=max_num_span)
            span_lens = span_padding(span_lens, value=0, max_len=max_num_span)
            morph_idxes = span_padding(morph_idxes, value=[0] * max_span_len, max_len=max_num_span)
            span_weights = span_padding(span_weights, value=0, max_len=max_num_span)
            span_token_idxes = span_padding(span_token_idxes, value=(0, 0), max_len=max_num_span)
            span_word_idxes = span_padding(span_word_idxes, value=(0, 0), max_len=max_num_span)

            # prepare bert input
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [tokenizer.sep_token]

            pad_len = max_seq_len - len(tokens)
            # Zero-pad up to the sequence length. [cls] + inputs + [sep] + pads
            input_ids = tokenizer.convert_tokens_to_ids(tokens + [tokenizer.pad_token] * pad_len)
            input_mask = [1] * len(tokens) + [0] * pad_len
            segment_ids = [sequence_a_segment_id] * len(tokens) + [pad_token_segment_id] * pad_len

            # 在 example 处理的时候，将 token 越界的 span 已经处理了
            # 原始 example 的 span 是不会越界的，所以其 cont_idx 是正常的
            # 替换后的 example, 因为可能的span越界，可能会减少 span 数量， 这种情况下，负例的 cont_span_idx 为 0
            # 这会对训练带来noise， 考虑到这种情况是少数，暂忽略不计
            # 这里 span 可能越界，因为word变成token之后会变多，原来对word合法的span，对token不再合法

            numeric_features.append(
                [input_ids, input_mask, segment_ids,
                 span_token_idxes, span_labels, span_weights,
                 morph_idxes, span_lens, span_masks, cont_span_idx,
                 span_word_idxes]
            )
        return numeric_features


def load_examples(data_dir, mode, tokenizer):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))

    if not os.path.exists(file_path):
        raise Exception("Cant find {} from given dir {}".format("{}.txt".format(mode), data_dir))
    guid_index = 1
    examples = []

    with open(file_path, encoding="utf-8") as f:
        words = []
        bio_labels = []

        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    # empty cached sample
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            bio_labels=bio_labels,
                            tokenizer=tokenizer
                        )
                    )
                    guid_index += 1
                    words = []
                    bio_labels = []
            else:
                # cache token and its bio format label
                splits = line.split()

                if len(splits) not in [2, 4]:
                    logger.warning("Line '{}' split num not in [2, 4], default '{}' as word and '{}' as label"
                                   .format(line, splits[0], splits[-1]))
                words.append(splits[0])

                if len(splits) > 1:
                    bio_labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    bio_labels.append("O")
        if words:
            examples.append(
                InputExample(
                    guid="{}-{}".format(mode, guid_index),
                    words=words,
                    bio_labels=bio_labels,
                    tokenizer=tokenizer
                )
            )
    return examples


def collate_fn(batch):
    """
    Convert batch samples to input tensors.

    Args:
        batch: ?

    Returns:
        original sample features, contrastive sample features

    """
    features = []

    for idx in range(len(['ori_feas', 'cont_feas'])):
        features.append({
            'input_ids': torch.LongTensor([sample[idx][0] for sample in batch]),
            'input_mask': torch.LongTensor([sample[idx][1] for sample in batch]),
            'segment_ids': torch.LongTensor([sample[idx][2] for sample in batch]),
            'span_token_idxes': torch.LongTensor([sample[idx][3] for sample in batch]),
            'span_labels': torch.LongTensor([sample[idx][4] for sample in batch]),
            'span_weights': torch.Tensor([sample[idx][5] for sample in batch]),
            'morph_idxes': torch.LongTensor([sample[idx][6] for sample in batch]),
            'span_lens': torch.LongTensor([sample[idx][7] for sample in batch]),
            'span_masks': torch.LongTensor([sample[idx][8] for sample in batch]),
            'cont_span_idx': torch.LongTensor([sample[idx][9] for sample in batch]),
            'span_word_idxes': torch.LongTensor([sample[idx][10] for sample in batch])
        })

    return features


def convert_2_token_idx(offsets, span_idxes, max_len=128):
    """
    convert the all the span_idxes from word-level to token-level
    """
    n_span_keep = 0
    span_idxes_ltoken = []

    for start, end in span_idxes:
        # avoid span right out of index, +1 for [CLS]
        if offsets[end][-1] > max_len - 1:
            continue
        n_span_keep += 1
        span_idxes_ltoken.append((offsets[start][0] + 1, offsets[end][-1] + 1))

    return span_idxes_ltoken, n_span_keep


def get_case_feature(span_idxes, words, max_span_len):
    """
    this function use to characterize the capitalization feature.
    Args:
        span_idxes:
        words:

    Returns:

    """
    caseidxes = []

    for idxes in span_idxes:
        sid, eid = idxes
        span_word = words[sid:eid + 1]
        caseidx1 = [0 for _ in range(max_span_len)]

        for j, token in enumerate(span_word):
            if token.isupper():
                tfeat = 'isupper'
            elif token.islower():
                tfeat = 'islower'
            elif token.istitle():
                tfeat = 'istitle'
            elif token.isdigit():
                tfeat = 'isdigit'
            else:
                tfeat = 'other'
            caseidx1[j] = morph2idx[tfeat]
        caseidxes.append(caseidx1)

    return caseidxes


def span_padding(lst, value=None, max_len=502):
    while len(lst) < max_len:
        lst.append(value)

    return lst[: max_len]


def get_labels(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        raise Exception("Cant find labels file: {}".format(path))
