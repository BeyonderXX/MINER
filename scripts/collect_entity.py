# coding: utf-8
import json


def get_entity(training_file, entity_out):
    fi = open(training_file, "r+", encoding='utf-8')
    # 保存实体词表
    vocab_dic = {}
    training_lines = fi.readlines()

    for segs_list in generate_sample(training_lines):
        labels = [segs[-1] for segs in segs_list]
        entities = get_entities_bio(labels)

        for entity in entities:
            entity_span = [segs[0] for segs in segs_list[entity[1]: entity[2]+1]]

            if entity[0] not in vocab_dic:
                vocab_dic[entity[0]] = ["<split>".join(entity_span)]
            else:
                if "<split>".join(entity_span) not in vocab_dic[entity[0]]:
                    vocab_dic[entity[0]].append("<split>".join(entity_span))

    feo = open(entity_out, "w+", encoding='utf-8')
    json.dump(vocab_dic, feo, ensure_ascii=False, indent=2)
    print('Finish entity span collect!')

    fi.close()
    feo.close()

    return vocab_dic



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
            chunk[0] = tag.replace("B-", "")
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.replace("I-", "")
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return set([tuple(chunk) for chunk in chunks])


if __name__ == "__main__":
    origin_data = '../data/WNUT2017/origin/train.txt'
    out_data = 'entity_old.json'
    get_entity(origin_data, out_data)
