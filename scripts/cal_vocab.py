import json
import math

from collections import OrderedDict
from transformers import AutoTokenizer


doc_str = "-DOCSTART-"


def get_entities_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return set([tuple(chunk) for chunk in chunks])


# 迭代式返回sample样本
def generate_sample(total_lines, start_index=0):
    segs_list = []

    for i in range(start_index, len(total_lines)):
        line = total_lines[i].strip()

        # segs 可能为空
        if line == "":
            if segs_list == []:
                continue
            else:
                yield segs_list
                segs_list = []
                continue

        segs = line.split()

        if len(segs) not in [2, 4]:
            print(i)
            raise Exception(
                "Error line {0} with length {1}".format(line, len(segs)))
        segs_list.append(segs)

    if segs_list:
        yield segs_list


def get_entity(training_file, entity_out):
    fi = open(training_file, "r+", encoding='utf-8')
    # 保存实体词表
    vocab_dic = {'O': []}
    training_lines = fi.readlines()

    for segs_list in generate_sample(training_lines):
        labels = [segs[-1] for segs in segs_list]
        entities = get_entities_bio(labels)

        for entity in entities:
            entity_span = [segs[0] for segs in segs_list[entity[1]: entity[2]+1]]

            if entity[0] not in vocab_dic:
                vocab_dic[entity[0]] = [' '.join(entity_span)]
            else:
                vocab_dic[entity[0]].append(' '.join(entity_span))

        for i, label in enumerate(labels):
            if label == 'O':
                vocab_dic['O'].append(segs_list[i][0])

    feo = open(entity_out, "w+", encoding='utf-8')
    json.dump(vocab_dic, feo, ensure_ascii=False, indent=2)

    fi.close()
    feo.close()

    return vocab_dic


# TODO 把 n>2 长度的 subword 都考虑进来
def calculate_PMI(entity_dic, tokenizer_path, out_path):
    """
    长度为 1 的 subword 计算
    PMI = log(p(x, y)/(p(x)p(y)))
    :return:
    """
    tokenizer = load_tokenizer(tokenizer_path)
    subword_dic = {}
    sum_subword_dic = {}

    for entity_type in entity_dic:
        subword_dic[entity_type] = {}

        for entity in entity_dic[entity_type]:
            for subword in tokenize(entity, tokenizer):
                if isinstance(subword, bytes):
                    subword = str(subword, encoding='utf-8')

                subword_dic[entity_type][subword] = subword_dic[entity_type].get(subword, 0) + 1
                sum_subword_dic[subword] = sum_subword_dic.get(subword, 0) + 1

    sum_count = sum(value for value in sum_subword_dic.values())
    PMI = {}

    for entity_type in subword_dic:
        pmi_dic = {}
        entity_type_count = sum(value for value in subword_dic[entity_type].values())
        py = entity_type_count / sum_count

        for subword in set(subword_dic[entity_type]):
            px = sum_subword_dic[subword] / sum_count
            pxy = subword_dic[entity_type][subword] / sum_count

            pmi_dic[subword] = math.log(pxy/(px*py))
        pmi_dic = [list(x) for x in sorted(pmi_dic.items(),
                                           key=lambda item: item[1],
                                           reverse=True)]
        PMI[entity_type] = pmi_dic

    fo = open(out_path, "w+", encoding='utf-8')
    json.dump(PMI, fo, ensure_ascii=False, indent=2)

    return PMI


def tokenize(token, tokenizer):
    return tokenizer.tokenize(token)


def load_tokenizer(tokenizer_path):
    print("Load tokenizer!!!")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=None
    )
    # x = ["abcdeengineroadsubwordaaauniversityoverrly",
    # "conll2003origindir", "FudanUniversity"]
    #
    # for word in x:
    #     print(tokenizer.tokenize(word))

    return tokenizer


def count_len_subword(pmi_json):
    len_dic = {}
    from prettyprinter import cpprint
    for type in pmi_json:
        type_len_dic = {}
        for subword in pmi_json[type]:
            subword = subword[0].replace('##', '')
            type_len_dic[len(subword)] = type_len_dic.get(len(subword), 0) + 1

        len_dic[type] = type_len_dic

    cpprint(len_dic)


if __name__ == '__main__':
    from prettyprinter import cpprint
    training_file = '../data/OpenNER/origin/train.txt'
    entity_out = 'entity_test.txt'
    entity_json = get_entity(training_file, entity_out)
    # # eo = open(entity_out, 'r+', encoding='utf-8')
    # # entity_json = json.load(eo)
    # # eo.close()
    pmi_out = 'pmi.txt'
    tokenizer_conf_path = '/root/MODELS/bert-base-uncased/'
    PMI = calculate_PMI(entity_json, tokenizer_conf_path, entity_out)
    #
    # ratio = 0.05
    #
    # for entity_type in PMI:
    #     token_pmi = PMI[entity_type]
    #     cpprint("*"*50)
    #     cpprint(entity_type)
    #     cpprint(token_pmi[:int(len(token_pmi)*ratio)])


    # calculate length
    tokenizer_conf_path = '/root/MODELS/bert-base-uncased/'
    load_tokenizer(tokenizer_conf_path)
    count_len_subword(PMI)
