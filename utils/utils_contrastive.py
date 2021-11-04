import os
import copy
import string
import random, json

from utils.typos import typos
from utils.utils_metrics import get_entities


# 为了保证 neg 里保留的 subword 会被重新切分出来，直接返回替换后subword的tokenize序列
def auto_neg_sample(
    origin_sample,
    total=False,
    mode='typos',
    tokenizer=None,
    pmi_filter=None,
    entity_json=None,
    switch_ratio=0.5
):
    if total:
        if mode == "typos":
            # TODO, add configuration
            if not entity_json and random.random() < switch_ratio:
                return entity_switch(origin_sample, tokenizer, entity_json)
            else:
                return total_rand_typos(origin_sample, tokenizer, pmi_filter)

        elif mode == "ngram":
            return total_rand_ngram(origin_sample, tokenizer, pmi_filter)
        else:
            raise NotImplementedError("Not support {} mode!".format(mode))
    else:
        if mode == "typos":
            # TODO, add configuration
            if not entity_json and random.random() < switch_ratio:
                return entity_switch(origin_sample, tokenizer, entity_json)
            else:
                return part_rand_typos(origin_sample, tokenizer, pmi_filter)

        elif mode == "ngram":
            return part_rand_ngram(origin_sample, tokenizer, pmi_filter)
        else:
            raise NotImplementedError("Not support {} mode!".format(mode))


def rand_replace(rep_fun):
    """
    实体词按照某种模式替换
    """
    def gen_neg_subwords(example, tokenizer, auxiliary_json=None):
        neg_example = copy.deepcopy(example)
        entities = list(get_entities(neg_example.labels))

        if not entities:
            return copy.deepcopy(example)

        # 默认一个sample只操作一个实体序列, switch_start switch_end 在 origin sample 中已经选定
        i, j = example.switch_start, example.switch_end
        label = example.labels[i]
        entity = [label, i, j]
        rep_words = rep_fun(example, entity, tokenizer, auxiliary_json)

        neg_example.words = neg_example.words[:i] + rep_words + \
                            neg_example.words[j+1:]

        # rep_labels = ["B-" + entity[0]] + ["I-" + entity[0]] * (len(rep_words) -1)
        rep_labels = [entity[0]] * len(rep_words)
        neg_example.labels = neg_example.labels[:i] + rep_labels + \
                             neg_example.labels[j+1:]
        # TODO, verify
        neg_example.switch_start = i
        neg_example.switch_end = i + len(rep_words) - 1

        assert len(neg_example.words) == len(neg_example.labels)

        return neg_example

    return gen_neg_subwords


@rand_replace
def entity_switch(example, entity, tokenizer, auxiliary_json=None):
    i, j = entity[1:]
    random_entity = random.choice(auxiliary_json[entity[0]])
    # TODO, add default value
    entity_words = random_entity.split("<split>")

    return entity_words


@rand_replace
def total_rand_ngram(example, entity, tokenizer, PMI_filter=None):
    """
    实体词全部替换为随机的ngram
    """
    i, j = entity[1:]
    ngrams = [generate_ngram() for _ in range(i, j+1)]

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

    return typos_list


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

    return new_words


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

    return new_words


def mask_rand_typos(example, entity, tokenizer, PMI_filter=None):
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

    return new_words


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



# 构建 typos 负例样本
# 暂时无用
def build_typos_neg_examples(
        examples,
        tokenizer,
        neg_total=False,
        neg_mode='typos',
        pmi_json=None,
        preserve_ratio=0.3,
        entity_json=None,
        switch_ratio=0.5
):
    pmi_filter = None

    if pmi_json:
        os.path.exists(pmi_json)
        pmi_json = json.load(open(pmi_json, "r+", encoding='utf-8'))
        pmi_filter = {k: v[:int(len(v) * preserve_ratio)] for k, v in
                      pmi_json.items()}

    neg_examples = []
    for (ex_index, example) in enumerate(examples):
        neg_examples.append(
            auto_neg_sample(
                example,
                total=neg_total,
                mode=neg_mode,
                tokenizer=tokenizer,
                pmi_filter=pmi_filter,
                entity_json=entity_json,
                switch_ratio=switch_ratio
            )
        )
    return neg_examples


# 构建 mask 样本
# 暂时无用
def build_ent_mask_examples(
        examples,
        tokenizer,
        mask_ratio=0.85,
        pmi_json=None,
        preserve_ratio=0.3,
        mask_token='[MASK]'
):
    pmi_filter = None
    if pmi_json:
        os.path.exists(pmi_json)
        pmi_json = json.load(open(pmi_json, "r+", encoding='utf-8'))
        pmi_filter = {k: v[:int(len(v) * preserve_ratio)] for k, v in
                      pmi_json.items()}

    masked_examples = []

    for index, example in enumerate(examples):
        entities = list(get_entities(example.labels))
        new_example = copy.deepcopy(example)
        random_num = random.uniform(0, 1)

        # if not entities and random_num < 0.5:
        if not entities:
            masked_examples.append(new_example)
            continue

        for entity in entities:
            i, j = entity[1:]
            random_num = random.uniform(0, 1)

            rep_words = [mask_token for _ in range(i, j + 1)]
            new_example.words = new_example.words[:i] + rep_words + \
                                new_example.words[j + 1:]

            # if random_num < mask_ratio:  # replace by [MASK]
            #     rep_words = [mask_token for _ in range(i, j+1)]
            #     new_example.words = new_example.words[:i] + rep_words + \
            #                         new_example.words[j + 1:]
            # elif random_num < mask_ratio + (1 - mask_ratio) / 2:  # replace with typos words
            #     rep_words = mask_rand_typos(new_example, entity, tokenizer,
            #                                 PMI_filter=pmi_filter)
            #
            #     new_example.words = new_example.words[:i] + rep_words + \
            #                         new_example.words[j + 1:]

        assert len(new_example.words) == len(new_example.labels)
        masked_examples.append(new_example)

    return masked_examples
