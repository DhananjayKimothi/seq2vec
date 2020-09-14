from collections import namedtuple
import re
import numpy as np
from Bio import SeqIO
import warnings
import time
import pickle as pkl
from MakeVocab1 import BuildVocab
from DocVecsArray import DocvecsArray
from numpy import random
from collections import defaultdict

file = "yeast_seq.fasta"
input_path = r"../data"
output_path = r"../data/inpfiles"# save folder

class seq_slicing(object):
    def __init__(self, w_s, w_type, n_w):
        """
        w_s is word size, w_type is for selecting how you want to
        slice the sequence: 'overlap' or 'non_overlap' ; n_w is special parameter
        required only when non overlap is used, ex: for w_s = 3, n_w = '...?'
        """
        self.w_s = w_s
        self.w_type = w_type
        self.n_w = n_w

    def slices(self, seq):
        if self.w_type == 'overlap':
            words = []
            for i in range(0, len(seq) - self.w_s + 1):
                words.append(seq[i:i + self.w_s])
            return words

        if self.w_type == 'non_overlap':
            words = re.findall(self.n_w, seq)
            seq_len = len(seq)
            p = seq_len // self.w_s  # floored quotient of seq_len
            words = words[0:p]
            seq1 = seq

            xx = np.zeros(self.w_s - 1)  # to delete 1st index, del seq1[i]
            xx = xx.astype(np.int)

            words_list = []
            words2 = []
            for j in xx:
                seq1 = list(seq1)
                del seq1[j]
                seq1 = "".join(seq1)
                seq_len = len(seq1)
                words1 = re.findall(self.n_w, seq1)
                p = seq_len // self.w_s
                words1 = words1[0:p]
                words2.extend(words1)
            words.extend(words2)
            return words


class TaggedDocument(namedtuple('TaggedDocument', 'words tags')):
    """
    A single document, made up of `words` (a list of unicode string tokens)
    and `tags` (a list of tokens). Tags may be one or more unicode string
    tokens, but typical practice (which will also be most memory-efficient) is
    for the tags list to include a unique integer id as the only tag.

    Replaces "sentence as a list of words" from Word2Vec.

    """

    def __str__(self):
        return '%s(%s,%s)' % (self.__class__.__name__,  self.words,  self.tags)
        # return '%s(%s,%s, %s)' % (self.__class__.__name__,   self.words,  self.tags)


class LabeledSentence(TaggedDocument):
    def __init__(self, *args, **kwargs):
        warnings.warn('LabeledSentence has been replaced by TaggedDocument', DeprecationWarning)


class LabeledLineSentence(seq_slicing):
    
    def __init__(self, filename,ids,w_s, w_type, n_w):
        super(LabeledLineSentence, self).__init__(w_s=w_s, w_type=w_type, n_w=n_w)
        self.filename = filename
        #self.SeqDict = SeqIO.to_dict(SeqIO.parse(filename, "fasta"))  # dictonary: keys as fasta ids
        self.fastaDict = defaultdict()
        self.ids = ids
    
    def read_fasta_file_as_dict(self):
        """
        It reads fasta file as a dictionary with
        keys as sequence IDs (without '>')
        and the value as a sequence (excluding "\n" if there is any)
        """
        IdsL = []
        # file_ = os.path.join(path,fileN)
        # file = open(file_,'r')
        lines_iter = iter(open(self.filename,'r'))
        for line in lines_iter:
            if line[0] == '>':
                seq = ''
                id = line.strip().replace('>','')
                self.fastaDict[id] = ''
                IdsL.append(id)
            else:
                self.fastaDict[id] += line.strip()

   
    
    def __iter__(self):
        self.read_fasta_file_as_dict()
        i = 0
        for key in self.ids:
            cls_lbl = key.split('_')
            key = key.split('\n')
            key = key[0]
            seq = self.fastaDict[key]
            #seq = seq_record.seq
            #seq = str(seq)
            kmer_list = self.slices(seq)  # word size
            #tag = seq_record.id # tag is key
            # yield TaggedDocument(cls_lbl[2], kmer_list,tags=[tag])
            yield TaggedDocument(kmer_list,tags=[key])
            # yield LabeledSentence(kmer_list, tags=['SEQ_%s' % i])
            i = i+1




def sequence_ids(file_train_p):
    IdsL = []
    file_train = open(file_train_p,'r')
    for line in file_train:
        if line[0] == '>':
            line = line.replace('>', '')
            line = line.replace('\r\n','')
            IdsL.append(line)
    file_train.close()

    return IdsL


class Seq2Vec(BuildVocab):
    def __init__(self,sequences,path_no,file_no):
        self.min_count = 0
        self.sample = 0
        self.max_vocab_size = 100000
        self.docvecs = DocvecsArray()
        self.hs = 1
        self.vector_size = 100  # check this --its a vector size
        self.layer1_size = self.vector_size
        self.negative = 0
        self.sorted_vocab = 0  # want to sort vocab on the basis of frequency ?
        self.null_word = 0  # check this
        self.dm = 1
        self.dm_concat = 0
        self.seed = 0
        self.random = random.RandomState(self.seed)
        self.hashfxn = hash
        self.total_words = 0
        self.build_vocab(sequences)  # the function is defined in BuildVocab
        self.path_no = path_no
        self.file_no = file_no
        """
        Saving initializations

        """

        # path = r"E:\SuperVec_Codes_25072018\Data\50_classes\DifferentDimension"


        x = self.file_no
        index2word_py = open(os.path.join(output_path, 'index2word' + str(x) + '.pkl'),'w')
        pkl.dump(self.index2word, index2word_py)

        vocab_py = open(output_path + '\\vocab' + str(x) + '.pkl', 'w')  # kmers, paths and codes
        pkl.dump(self.vocab, vocab_py)

        doctag_py = open(output_path + '\doctag' + str(x) + '.pkl',
                         'w')  # doctag initialization
        pkl.dump(self.docvecs.doctag_syn0, doctag_py)

        kmer_py = open(output_path + '\kmer' + str(x) + '.pkl', 'w')  # kmer initialization
        pkl.dump(self.syn0, kmer_py)

        vocab_py.close()
        doctag_py.close()
        kmer_py.close()
        index2word_py.close()



def main(path_no,file_no):

    word_size = 3  # size of each word
    window_type = 'non_overlap'  # can choose overlap or non overlap
    n_w = '...?'  # used when non overlaping window is selecte
    filepath = os.path.join(input_path,file)
    print(filepath)


    IdsL = sequence_ids(filepath)
    # print(IdsL)
    seq_corpus = LabeledLineSentence(filepath, IdsL, w_s=word_size, w_type=window_type, n_w=n_w)

    Seq2Vec(seq_corpus,path_no,file_no)


if __name__ == '__main__':
    import sys
    import os
    path_no = 0
    for i in range(0,1):
        file_no = i
        print("here")
        main(path_no,file_no)


### Run vocab_text.py
""""

system arguments ---- path and file number

"""
