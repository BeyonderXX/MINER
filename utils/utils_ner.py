import logging
import os
import torch
import string
import random
import copy
import json

from utils.typos import typos
from utils.utils_metrics import get_entities

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence.
            This should be specified for train and dev examples,
            but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, valid_mask, segment_ids,
                 label_ids, neg_input_ids, neg_input_mask,
                 neg_valid_mask, neg_segment_ids
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.valid_mask = valid_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.neg_input_ids = neg_input_ids
        self.neg_valid_mask = neg_valid_mask
        self.neg_input_mask = neg_input_mask
        self.neg_segment_ids = neg_segment_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                if len(splits) not in [2, 4]:
                    print(line)

                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
    return examples


def get_valid_masks(tokenized_subwords):
    valid_mask = []
    tokens = []

    for subwords in tokenized_subwords:
        # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
        for i, word_token in enumerate(subwords):
            if i == 0:
                valid_mask.append(1)
            else:
                valid_mask.append(0)
            tokens.append(word_token)

    return tokens, valid_mask


# 将 example 改为模型的输入，当输入长度tokenize之后超过最大长度时，evaluate 可能会报错
def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        neg_total=False,
        neg_mode='typos',
        pmi_json=None,
        preserve_ratio=0.3
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    pmi_filter = None

    if pmi_json:
        os.path.exists(pmi_json)
        pmi_json = json.load(open(pmi_json, "r+", encoding='utf-8'))
        pmi_filter = {k: v[:int(len(v)*preserve_ratio)] for k, v in pmi_json.items()}

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokenized_words = []

        for word in example.words:
            tokenized_words.append(tokenizer.tokenize(word))
        tokens, valid_mask = get_valid_masks(tokenized_words)

        feature = example_to_feature(
            tokenized_words, example.labels, label_list, max_seq_length, tokenizer,
            cls_token_at_end=cls_token_at_end,
            cls_token=cls_token,
            cls_token_segment_id=cls_token_segment_id,
            sep_token=sep_token,
            sep_token_extra=sep_token_extra,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,
            pad_token_label_id=pad_token_label_id,
            sequence_a_segment_id=sequence_a_segment_id,
            mask_padding_with_zero=mask_padding_with_zero
        )
        input_ids, input_mask, valid_mask, segment_ids, label_ids = feature
        check_feature(*feature, max_seq_length)

        # 假设为等长替换
        neg_tokenized_words = auto_neg_sample(
            example,
            total=neg_total,
            mode=neg_mode,
            tokenizer=tokenizer,
            pmi_filter=pmi_filter
        )
        neg_tokens, neg_valid_mask = get_valid_masks(neg_tokenized_words)
        # TODO modify labels
        neg_feature = example_to_feature(
            neg_tokenized_words, example.labels, label_list, max_seq_length, tokenizer,
            cls_token_at_end=cls_token_at_end,
            cls_token=cls_token,
            cls_token_segment_id=cls_token_segment_id,
            sep_token=sep_token,
            sep_token_extra=sep_token_extra,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,
            pad_token_label_id=pad_token_label_id,
            sequence_a_segment_id=sequence_a_segment_id,
            mask_padding_with_zero=mask_padding_with_zero
        )
        neg_input_ids, neg_input_mask, neg_valid_mask, neg_segment_ids, neg_label_ids = neg_feature
        check_feature(*neg_feature, max_seq_length)

        if ex_index < 1:
            log_example(tokens, *feature)
            log_example(neg_tokens, *neg_feature, prefix='negative')

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          valid_mask=valid_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,

                          neg_input_ids=neg_input_ids,
                          neg_input_mask=neg_input_mask,
                          neg_valid_mask=neg_valid_mask,
                          neg_segment_ids=neg_segment_ids
                          )
        )
    return features


def log_example(tokens, input_ids, input_mask, valid_mask,
                segment_ids, label_ids, prefix=''):
    logger.info("*** Example {}***".format(prefix))
    logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
    logger.info("valid_mask: %s", " ".join([str(x) for x in valid_mask]))
    logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
    logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
    logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
    logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))


def check_feature(input_ids, input_mask, valid_mask,
                  segment_ids, label_ids, max_seq_length):
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(valid_mask) == max_seq_length


def example_to_feature(
        tokenized_words,
        labels,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True
):
    label_map = {label: i for i, label in enumerate(label_list)}
    tokens, valid_mask = get_valid_masks(tokenized_words)
    label_ids = [label_map[label] for label in labels]

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]
        valid_mask = valid_mask[: (max_seq_length - special_tokens_count)]

    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    valid_mask.append(1)

    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        valid_mask.append(1)

    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        label_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
        valid_mask.append(1)
    else:
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids
        valid_mask.insert(0, 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        label_ids = ([pad_token_label_id] * padding_length) + label_ids
        valid_mask = ([0] * padding_length) + valid_mask
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        valid_mask += [0] * padding_length

    while (len(label_ids) < max_seq_length):
        label_ids.append(pad_token_label_id)

    return [input_ids, input_mask, valid_mask, segment_ids, label_ids]


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    batch_tuple = tuple(map(torch.stack, zip(*batch)))
    batch_lens = torch.sum(batch_tuple[1], dim=-1, keepdim=False)
    max_len = batch_lens.max().item()
    results = ()
    for item in batch_tuple:
        if item.dim() >= 2:
            results += (item[:, :max_len],)
        else:
            results += (item,)
    return results


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER",
                "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


# 为了保证 neg 里保留的 subword 会被重新切分出来，直接返回替换后subword的tokenize序列
def auto_neg_sample(
    origin_sample,
    total=False,
    mode='typos',
    tokenizer=None,
    pmi_filter=None
):
    if total:
        if mode == "typos":
            return total_rand_typos(origin_sample, tokenizer, pmi_filter)
        elif mode == "ngram":
            return total_rand_ngram(origin_sample, tokenizer, pmi_filter)
        else:
            raise NotImplementedError("Not support {} mode!".format(mode))
    else:
        if mode == "typos":
            return part_rand_typos(origin_sample, tokenizer, pmi_filter)
        elif mode == "ngram":
            return part_rand_ngram(origin_sample, tokenizer, pmi_filter)
        else:
            raise NotImplementedError("Not support {} mode!".format(mode))


def build_neg_samples(
    origin_samples,
    total=True,
    rep_freq=False,
    mode='typos',
    tokenizer=None,
    pmi_path=None,
    pmi_preserve=0.1
):
    """
    1. 怎么选取被替换实体词（1> 替换部分还是全部替换； 2> 按照频率替换，还是随机选择）
    2. 怎么选取替换词 （1> ngram; 2> Typos; 3> 随机词表词 4> HotFlip rank ）

    目前为等长替换策略

    """
    if rep_freq:
        raise NotImplementedError("Not implement replace frequency entity!")

    if total:
        if mode == "typos":
            return [total_rand_typos(x) for x in origin_samples]
        elif mode == "ngram":
            return [total_rand_ngram(x) for x in origin_samples]
        else:
            raise NotImplementedError("Not support {} mode!".format(mode))
    else:
        assert pmi_path is not None
        pmi_json = json.load(open(pmi_path, "r+", encoding='utf-8'))
        pmi_filter = {k: v[:int(len(v)*pmi_preserve)] for k, v in pmi_json.items()}
        if mode == "typos":
            return [part_rand_typos(x, tokenizer, pmi_filter) for x in origin_samples]
        elif mode == "ngram":
            return [part_rand_ngram(x, tokenizer, pmi_filter) for x in origin_samples]
        else:
            raise NotImplementedError("Not support {} mode!".format(mode))


def rand_replace(rep_fun):
    """
    实体词按照某种模式替换
    """
    def gen_neg_subwords(example, tokenizer, PMI_filter=None):
        new_example = copy.deepcopy(example)
        entities = list(get_entities(new_example.labels))

        if not entities:
            return copy.deepcopy(example.words)

        # 默认一个sample只操作一个实体序列
        entity = random.choice(entities)
        i, j = entity[1:]
        rep_sub_words = rep_fun(example, entity, tokenizer, PMI_filter=PMI_filter)

        new_sub_words = [tokenizer.tokenize(x) for x in new_example.words[:i]] \
                        + rep_sub_words + \
                        [tokenizer.tokenize(x) for x in new_example.words[j+1:]]

        assert len(new_sub_words) == len(new_example.labels)

        return new_sub_words

    return gen_neg_subwords


@rand_replace
def total_rand_ngram(example, entity, tokenizer, PMI_filter=None):
    """
    实体词全部替换为随机的ngram
    """
    i, j = entity[1:]
    ngrams = [tokenizer.tokenize(generate_ngram()) for _ in range(i, j+1)]

    return ngrams


@rand_replace
def total_rand_typos(example, entity, tokenizer, PMI_filter=None):
    """
    实体词全部加入Typos

    """
    i, j = entity[1:]
    typos_list = []

    for idx in range(i, j + 1):
        candidate = typos.get_candidates(example.words[idx], n=1)
        if candidate:
            candidate = candidate[0]
        else:
            candidate = example.words[idx]
        typos_list.append(candidate)

    return [tokenizer.tokenize(word) for word in typos_list]


@rand_replace
def part_rand_ngram(example, entity, tokenizer, PMI_filter=None):
    """
    只用 ngram 替换 PMI 值低的 subword

    """
    i, j = entity[1:]
    new_words = []

    for token in example.words[i: j+1]:
        subwords = tokenizer.tokenize(token)
        new_subwords = []

        for subword in subwords:
            # sub word 和 PMI_filer 都带有 ## 表示
            if subword in PMI_filter[entity[0]]:
                new_subwords.append(subword)
            else:
                # 判断 ##
                rep_subword = generate_ngram()
                if subword[:2] == '##':
                    rep_subword = '##' + rep_subword
                new_subwords += rep_subword
        # 如果将tokenize的subword加入subwords，会极大地加长subword的数目
        new_words.append(reverse_BERT_tokenize(new_subwords))

    return [tokenizer.tokenize(word) for word in new_words]


@rand_replace
def part_rand_typos(example, entity, tokenizer, PMI_filter=None):
    """
    只向 PMI 值低的 subword 中插入 typos

    """
    i, j = entity[1:]
    new_words = []

    for token in example.words[i: j+1]:
        subwords = tokenizer.tokenize(token)
        # 保存的是一个词对应的subword
        new_subwords = []

        for subword in subwords:
            if subword in PMI_filter[entity[0]]:
                new_subwords.append(subword)
            else:
                # 判断 ##
                sub_wo_sig = subword[2:] if '##' == subword[:2] else subword
                rep_subwords = typos.get_candidates(sub_wo_sig, n=1)

                if rep_subwords:
                    rep_subword = rep_subwords[0]
                else:
                    rep_subword = sub_wo_sig

                if subword[:2] == '##':
                    rep_subword = '##' + rep_subword
                new_subwords.append(rep_subword)
        new_words.append(reverse_BERT_tokenize(new_subwords))

    assert(len(new_words) == j-i+1)

    return [tokenizer.tokenize(word) for word in new_words]


def reverse_BERT_tokenize(segs):
    """
    将BERT对单个词的tokenize结果还原
    :param segs:
    :return:
    """
    text = ' '.join([x for x in segs])
    return text.replace(' ##', '')


def generate_ngram(seed=42):
    chars = string.ascii_letters
    ngram_len = random.randint(2, 5)
    random_str = ''.join([random.choice(chars) for i in range(ngram_len)])

    return random_str


if __name__ == "__main__":
    examples = [InputExample(1, ["I", "lives", "in", "Shanghai", "Yangpu"],
                             ["O", "O", "O", "B-LOC", "I-LOC"]),
                InputExample(2, ["China", "contains", "Shanghai", "City"],
                             ["B-LOC", "O", "B-LOC", "I-LOC"]),
                InputExample(3, ["Bao", "Rong", "is", "playing", "LOL"],
                             ["B-PER", "I-PER", "O", "O", "O"])
                ]
    # start test
    for mode in ['typos', "ngram"]:
        print("#"*50)
        print(mode)
        trans_examples = build_neg_samples(examples, mode=mode)

        for example in trans_examples:
            print("*"*50)
            print(" ".join(example.words))
            print(" ".join(example.labels))
