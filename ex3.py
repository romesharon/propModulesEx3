from collections import Counter
from datetime import datetime
import pickle
import numpy as np

RARE_WORD = 3
EPSILON = 0.1
LAMBDA = 1
K = 5
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
                self.__n_t_k[document_index][self.__words[word]] = self.__parse_corpus[document][1].count(word)
            self.__n_t[document_index] = len(self.__parse_corpus[document][1])

        self.__w_t_i = np.zeros((len(self.__parse_corpus), len(self.__clusters)))
        for index, _ in enumerate(self.__parse_corpus):
            self.__w_t_i[index][index % len(self.__clusters)] = 1

        self.m_step()

    def m_step(self):
        self.calculate_alpha()
        self.__p_i_k = (np.dot(self.__w_t_i.T, self.__n_t_k) + LAMBDA) / (
                np.dot(self.__w_t_i.T, self.__n_t) + len(self.__words) * LAMBDA)

    def calculate_alpha(self):
        self.__alpha = np.sum(self.__w_t_i, axis=0) / len(self.__parse_corpus)
        self.__alpha = np.maximum(self.__alpha, EPSILON)
        self.__alpha /= np.sum(self.__alpha)

    def e_step(self, z, m):
        e_mat = np.where(z - m < (-1) * K, 0, np.exp(z - m))
        e_sum = np.sum(e_mat, axis=1)
        self.__w_t_i = e_mat / np.column_stack([e_sum for _ in range(e_mat.shape[1])])
        return

    def calculate_z(self):
        first = np.dot(self.__n_t_k, np.log(self.__p_i_k.T))
        second = np.broadcast_to(np.log(self.__alpha), (len(self.__parse_corpus), len(self.__clusters)))
        return first + second

    def calculate_m(self, z):
        return np.max(z, axis=1).reshape(len(self.__parse_corpus), 1)

    def train(self):
        current = np.PINF
        prev = np.PINF
        check_guess = []
        check_underflow_e = []
        i = 0
        while prev - current > EPSILON or current == np.PINF:
            print(i)
            print(current)
            i += 1
            z = self.calculate_z()
            m = self.calculate_m(z)
            self.e_step(z, m)
            self.m_step()
            prev = current

            # calculate log likelihood
            underflow_e = self.underflow_e_step(z, m)
            current = np.power(np.e, (-1 / np.sum(self.__n_t)) * underflow_e)

            check_underflow_e.append(underflow_e)
            check_guess.append(current)
        print(check_guess)
        print(check_underflow_e)

    def underflow_e_step(self, z, m):
        result = np.zeros((len(self.__parse_corpus), 1))
        for t, _ in enumerate(self.__parse_corpus):
            for i, _ in enumerate(self.__clusters):
                if z[t][i] - m[t] >= K * -1:
                    result[t] += np.exp(z[t][i] - m[t])
        return np.sum(np.log(result) + m)

    def calculate_accuracy(self):
        max_vector = np.argmax(self.__w_t_i, axis=1)
        cluster2topic = self.get_number_of_topics_per_cluster(max_vector)
        correct = 0
        for document_index, document in enumerate(self.__parse_corpus):
            topic = cluster2topic[max_vector[document_index]]
            if topic in self.__parse_corpus[document][0]:
                correct += 1
        return correct / len(self.__parse_corpus)

    def get_number_of_topics_per_cluster(self, max_vector):
        cluster2topic = {i: Counter() for i, _ in enumerate(self.__clusters)}
        for document_index, document in enumerate(self.__parse_corpus):
            cluster2topic[max_vector[document_index]].update(self.__parse_corpus[document][0])

        cluster2most_common_topic = dict()
        for cluster_index in cluster2topic:
            most_common_topic = cluster2topic[cluster_index].most_common(1)
            cluster2most_common_topic[cluster_index] = most_common_topic[0][0] if len(most_common_topic) > 0 else None

        return cluster2most_common_topic


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
        em.train()
        print(em.calculate_accuracy())
    # with open(CORPUS, 'r') as file:
    #     content = file.read()
    #     rare_words = get_rare_words(content)
    #     corpus = split_text_to_subjects(content, rare_words)
    #     print(datetime.now() - start)
    #     results = open(RESULTS, 'wb')
    #     pickle.dump(corpus, results)
    #     results.close()
