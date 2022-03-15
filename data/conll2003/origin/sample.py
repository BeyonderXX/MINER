import random

file_list = ['dev','train']
sentense_list = []
precentage = [0.1, 0.3, 0.5, 0.7]

for j in precentage:
    for i in file_list:
        sentense_list = []
        with open(i + '.txt', 'r') as f:
            word_list = []
            current_word = f.readline()
            while current_word != '':
                while current_word != '\n':
                    word_list.append(current_word)
                    current_word = f.readline()
                current_word = f.readline()
                sentense_list.append(word_list)
                word_list = []
            f.close()
        # print(i + '.txt have ' + str(len(sentense_list)) + 'words')
        # print(str(j * 100) + '%' + i + '.txt have ' + str(int(float(len(sentense_list)) * j)) + 'words')
        random.shuffle(sentense_list)
        print(str(j * 100) + i + '.txt')
        with open(str(int(j * 100)) + i + '.txt', 'w') as sample:
            for k in range(int(float(len(sentense_list)) * j)):
                for o in range(len(sentense_list[k])):
                    sample.write(sentense_list[k][o])
                sample.write("\n")
