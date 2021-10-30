# coding: utf-8

import random

def reading_file(path):
    with open(path, 'r+', encoding='utf-8') as fi:
        line = fi.readline()

        while(line):
            yield line.strip()
            line = fi.readline()


def open2conll(path, out_path):
    sample = []
    idx = 0
    fo = open(out_path + 'full.txt', "w+", encoding='utf-8')
    f_train = open(out_path+ 'train.txt', "w+", encoding='utf-8')
    f_dev = open(out_path+ 'dev.txt', "w+", encoding='utf-8')
    f_test = open(out_path+ 'test.txt', "w+", encoding='utf-8')
    count, train_count, dev_count, test_count = 0, 0, 0, 0


    for line in reading_file(path):
        line = line.strip()

        if line:
            sample.append(line)
        else:
            if sample:
                random_num = random.uniform(0, 1)
                conll_format = sample2conll(sample, idx)
                fo.write(conll_format)
                count += 1

                if random_num > 0.9 and train_count < 10000:
                    f_train.write(conll_format)
                    train_count += 1
                elif random_num > 0.9 and dev_count < 2000:
                    f_dev.write(conll_format)
                    dev_count += 1
                elif random_num > 0.9 and test_count < 2000:
                    f_test.write(conll_format)
                    test_count += 1

            sample = []
        idx += 1
    print(count, train_count, dev_count, test_count)
    fo.close()


def sample2conll(sample, idx):
    if len(sample) == 3:
        labels_info = sample[2].split("|")
    elif len(sample) == 2:
        print("idx num {0}".format(idx))
        print(sample)
        print("*"*50)
        labels_info = None

    else:
        print("idx num {0}".format(idx))
        print(sample)
        raise Exception('Error data/code!')


    content = sample[0]
    words = content.split(" ")
    labels= ['O'] * len(words)
    conll_content = ''

    if labels_info:
        for label_info in labels_info:
            segs = label_info.strip().split(" ")
            assert len(segs) == 2
            entity = segs[1]
            start, end = segs[0].split(',')
            labels[int(start)] = 'B-' + entity

            for j in range(int(start)+1, int(end)):
                labels[j] = 'I-' + entity

    for j in range(len(words)):
        conll_content += words[j] + ' ' + labels[j] + '\n'

    return conll_content + '\n'


if __name__ == "__main__":
    origin_data = '/Users/wangxiao/code/python/RobustNER/data/OpenNER/movie_song_book_tv_noquote.data'
    out_dir = '/Users/wangxiao/code/python/RobustNER/data/OpenNER/'
    open2conll(origin_data, out_dir)
