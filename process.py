from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import string
import fasttext
import numpy as np
from scipy import stats
import os
import math


def Normalize_data(input_path, output_path, lang):
    '''
    Return the corpus by normalising and removing the punctuations
    
    :param input_path: path to input raw corpus
    :type input_path: str
    :param output_path: path where the processed corpus will get stored
    :type output_path: str
    :param lang: corpus language
    :type lang: str
    '''
    punctuations = string.punctuation + r'‘’„“”–—´‘‚’'
    
    normalizer_factory = IndicNormalizerFactory()
    normalizer = normalizer_factory.get_normalizer(lang)

    def process_sent(sent):
        '''
        return the sentence after normalisation and removal of the punctuations
        
        :param sent: single line of words
        :type sent: str
        
        :return normalized: processed sentence
        :rtype normalised: str
        '''
        normalized = normalizer.normalize(sent)
        return normalized.translate(str.maketrans('', '', punctuations))

    with open(input_path, 'r', encoding='utf-8') as in_fp, open(output_path, 'w', encoding='utf-8') as out_fp:
        for line in in_fp.readlines():
            sent = line.rstrip('\n')
            toksent = process_sent(sent)
            out_fp.write(toksent)
            out_fp.write('\n')
           
        
def print_n_lines_from_file(file_path, n):
    '''
    prints n lines of text from the input file
    
    :param file_path: path to file from which the lines to be printed
    :type file_path: str
    :param n: number of lines to be printed
    :type n: int
    '''
    with open(file_path, 'r', encoding='utf-8') as in_fp:
        for i in range(n):
            line = in_fp.readline()
            print(line)
            
        
def train_n_models(n):
    '''
    train and save "n" fastext models
    
    :param n: number of models to be trained
    :type n: int
    '''
    for i in range(n):
        model = fasttext.train_unsupervised('/data1/hcs207019/data/mr/processed/mr.norm_punc_remv.txt', 
                                            model='skipgram', lr=0.09, dim=50, ws=10, epoch=2, minn=3, 
                                            maxn=6)
        file_name = '/data1/hcs207019/data/mr/models/fil' + str(i) + '.bin'
        model.save_model(file_name)
        
        
def proc_wordSim_dataset(input_path, output_path):
    '''
    Process the word similarity dataset (keep only the word pairs those have similarity score available)
    
    :param input_path: path to the row word similarity dataset
    :type input_path: str
    :param output_path: path to which the processed dataset need to be stored
    :type output_path: str
    '''
    def process_sim(line):
        '''
        return True  if current word pair have similarity score else return False
        
        :param line: single line containing word pair and similarity score
        :type line: str
        
        :return: True or False
        :rtype: Boolean
        '''
        s_line = line.split()
        if len(s_line) == 3:
            try:
                float(s_line[-1])
                return True
            except:
                return False
        else:
            return False

    with open(input_path, 'r', encoding='utf-8') as in_fp, open(output_path, 'w', encoding='utf-8') as out_fp:
        for line in in_fp.readlines():
            sent = line.rstrip('\n')
            if process_sim(sent):
                out_fp.write(sent)
                out_fp.write('\n')
                
                
def compute_similarity(data_path, model):
    '''
    returns the correlation score of the word embeddings (model) on the word similarity dataset along with the dataset name and the OOV percentage
    
    :param data_path: path at which the word similarity dataset is present
    :type: data_path: str
    :param model: word embedding matrix
    :type model: matrix of shape (n_words, embedding_dim)
    
    :return dataset: name of the dataset:
    :rtype dataset: str
    :return correlation: correlation score
    :rtyep correlation: float
    :return oov: Out Of Vocab percentage
    :rtyep oov: float
    '''
    def similarity(v1, v2):
        '''
        return the cosine similarity between two vectors
        
        :param v1: input vector 1
        :type v1: ndarray
        :param v2: input vector 2
        :type v2: ndarray
        
        :return: cosine similarity
        :rtype: float
        '''
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        return np.dot(v1, v2) / n1 / n2

    mysim = []
    gold = []

    with open(data_path, 'rb') as fin:
        for line in fin:
            tline = line.split()
            word1 = tline[0].lower()
            word2 = tline[1].lower()

            v1 = model.get_word_vector(word1)
            v2 = model.get_word_vector(word2)
            d = similarity(v1, v2)
            mysim.append(d)
            gold.append(float(tline[2]))

    corr = stats.spearmanr(mysim, gold)
    dataset = os.path.basename(data_path)
    correlation = corr[0] * 100
    return dataset, correlation, 0


def find_similarity_for_n_models(n):
    '''
    loads the "n" models and print their correlation scores
    
    :param n: nuber of models
    :type n: int
    
    :return models: list of "n" loaded models
    :rtype models: ndarray
    '''
    models = []
    for i in range(n):
        model_path = '/data1/hcs207019/data/mr/models/fil' + str(i) + '.bin'
        data_path ='/data1/hcs207019/data/mr/processed/proc_Marathi-WS.txt'
    
        model = fasttext.load_model(model_path)
        models.append(model)
        dataset, corr, oov = compute_similarity(data_path, model)
        print("{0:20s}: {1:2.0f}  (OOV: {2:2.0f}%)".format(dataset, corr, 0))
        print("\n")
        
    return models


def get_r_embbeding_matrix(models, r, words):
    '''
    return the list of r embbeding matrices
    
    :param models: list of fasttext models
    :type models: list
    :param r: number of models
    :type r: int
    :param words: list of all the words avaialble in the vocab
    :type words: list
    
    :return W: list of r embbeding matrices
    :rtype W: ndarray
    '''
    W = []
    for i in range(r):
        embb_matrix = [models[i][j] for j in words]
        W.append(embb_matrix)
    
    W = np.array(W)
    
    return W


def custom_compute_similarity(data_path, model, words):
    '''
    return correlation score for custom models on the word similarity dataset along with dataset name and the oov percentage
    
    :param data_path: path at which word similarity dataset is stored
    :type data_path: str
    :param model: word embbeding matrix
    :type model: ndarray
    :param words: list of all the words in vocab
    
    :return dataset: word similairty dataset name
    :rtype dataset: str
    :return correlation: correlation score
    :rtype: float
    return oov: oov percentage
    :rtype: flaot
    '''
    def similarity(v1, v2):
        '''
        return the cosine similarity between two vectors
        
        :param v1: input vector 1
        :type v1: ndarray
        :param v2: input vector 2
        :type v2: ndarray
        
        :return: cosine similarity
        :rtype: float
        '''
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        return np.dot(v1, v2) / n1 / n2

    mysim = []
    gold = []

    with open(data_path, 'r', encoding='utf-8') as fin:
        count = 0
        oov_count = 0
        x = 0
        for line in fin:
            tline = line.split()
            word1 = tline[0]
            word2 = tline[1]

            if (word1 in words) and (word2 in words):
                v1 = model[words.index(word1)]
                v2 = model[words.index(word2)]
                d = similarity(v1, v2)
                x += 1
            else:
                d = 0
                oov_count += 1
                
            count += 1    
            mysim.append(d)
            gold.append(float(tline[2]))
        
    corr = stats.spearmanr(mysim, gold)
    dataset = os.path.basename(data_path)
    correlation = corr[0] * 100
    return dataset, correlation, (oov_count*100.0)/count