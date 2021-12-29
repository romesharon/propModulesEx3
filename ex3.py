from collections import Counter
from datetime import datetime
import pickle

import numpy as np

RARE_WORD = 3
EPSILON = 0.000001
LAMBDA = 2
CORPUS = "dataset/develop.txt"
RESULTS = "results.pkl"
CLUSTERS = "dataset/topics.txt"


class EM(object):
    def __init__(self, parse_corpus, clusters):
        self.__clusters = clusters
        self.__parse_corpus = parse_corpus
        self.__words = {special_word: key for key, special_word in
                        enumerate(set([word for value in parse_corpus.values() for word in value[1]]))}
        self.__n_t = np.zeros((len(self.__parse_corpus), 1))
        self.__n_t_k = np.zeros((len(self.__parse_corpus), len(self.__words)))
        self.__p_i_k = np.zeros((len(self.__clusters), len(self.__words)))
        self.__alpha = np.zeros((len(self.__clusters), 1))
        self.nt = np.zeros((len(self.__parse_corpus), 1))

        for document_index, document in enumerate(self.__parse_corpus):
            for word in self.__parse_corpus[document][1]:
                self.__n_t_k[document_index][self.__words[word]] += 1
            self.__n_t[document_index] = len(self.__parse_corpus[document][1])

        self.__w_t_i = np.zeros((len(self.__parse_corpus), len(self.__clusters)))
        for index, _ in enumerate(self.__parse_corpus):
            self.__w_t_i[index][index % len(self.__clusters)] = 1

        self.m_step()

    def m_step(self):
        self.calculate_alpha()
        self.__p_i_k = (np.dot(self.__w_t_i.T, self.__n_t_k) + LAMBDA) / (
                np.dot(self.__w_t_i.T, self.__n_t) + len(self.__parse_corpus) * LAMBDA)

    def calculate_alpha(self):
        self.__alpha = np.sum(self.__w_t_i) / len(self.__parse_corpus)
        self.__alpha = np.maximum(self.__alpha, EPSILON)
        self.__alpha /= np.sum(self.__alpha)

    def e_step(self):
        z = self.z_matrix = self.calculate_z()
        m = self.m_matrix = np.max(z, axis=1).reshape(len(self.__parse_corpus), 1)
        for i in range(m.shape[0]):
            for j in range(z.shape[1]):
                z[i][j] -= m[i]
        z = z / 4
        e_mat = np.where(z < (-1) * 10, np.exp(z), np.exp(z))
        e_sum = np.sum(e_mat, axis=1)
        for i in range(e_mat.shape[0]):
            e_mat[i] /= e_sum[i]
        print(e_mat)
        self.__w_t_i = e_mat

    def calculate_z(self):
        first = np.dot(self.__n_t_k, np.log(self.__p_i_k.T))
        second = np.broadcast_to(np.log(self.__alpha), (len(self.__parse_corpus), len(self.__clusters)))
        return first + second

    # def train(self):
    #     prev_prep = float('inf')
    #     cur_prep = float('inf')
    #     while prev_prep - cur_prep > EPSILON or cur_prep == float('inf'):
    #         self.e_step()
    #         self.m_step()
    #         # print("acc", self.accuracy())
    #         print("likelihood", self.log_likelihood())
    #         prev_prep = cur_prep
    #         cur_prep = self.perplexity()
    #         print("cur_prep", cur_prep)
    #
    # def log_likelihood(self):
    #     boolean_table = self.z_matrix - self.m_matrix >= (-1) * 10
    #     array = np.zeros((len(self.__parse_corpus), 1))
    #     for t in range(len(self.__parse_corpus)):
    #         for i in range(len(self.__clusters)):
    #             if boolean_table[t][i]:
    #                 array[t] += np.exp(self.z_matrix[t][i] - self.m_matrix[t])
    #     return np.sum(np.log(array) + self.m_matrix)
    #
    # def perplexity(self):
    #     log_likelihood = self.log_likelihood()
    #     return np.power(2, -1 * log_likelihood / np.sum(self.nt))
    #


def get_rare_words(content):
    counting_words = Counter((' '.join(content.split('\n\n')[1::2])).split(' '))
    return [word for word in counting_words if counting_words[word] <= RARE_WORD]  # prints [5]


def extract_clusters(line: str):
    splited_line = line[1:-1].split('\t')
    return int(splited_line[1]), splited_line[2:]


def import_clusters():
    with open(CLUSTERS, 'rt') as f:
        data = f.read().split("\n\n")
        return {cluster: key for key, cluster in enumerate(data)}


def split_text_to_subjects(content, rare_words):
    index2word = dict()
    lines = content.split('\n\n')
    for i in range(0, (len(lines) - 2), 2):
        index, clusters = extract_clusters(lines[i])
        filterd_line = [word for word in lines[i + 1].split(' ') if word not in rare_words]
        index2word[index] = (clusters, filterd_line)
    return index2word


if __name__ == "__main__":
    start = datetime.now()
    with open(RESULTS, 'rb') as f:
        loaded_dict = pickle.load(f)
        clusters = import_clusters()
        em = EM(loaded_dict, clusters)
    # with open(CORPUS, 'r') as file:
    #     content = file.read()
    #     rare_words = get_rare_words(content)
    #     corpus = split_text_to_subjects(content, rare_words)
    #     print(datetime.now() - start)
    #     results = open(RESULTS, 'wb')
    #     pickle.dump(corpus, results)
    #     results.close()
