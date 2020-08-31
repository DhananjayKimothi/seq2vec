import pickle as pkl

import logging, os
logging.basicConfig(level=logging.DEBUG)
#os.remove("output.log")
logger = logging.getLogger("output")
logger.propagate = False

handler = logging.FileHandler("output.log")
handler.setLevel(logging.DEBUG)

# define formatter
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # stores the information for each log

# set formatter to the handler
#handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)





"""
def txt_prnt(file, value):
    #global vocab_txt
    last = 0
    for item in value:
        last += 1
        file.write(str(item))
        if last < len(value) :
            file.write(" ")
        else:
            file.write("\n")
"""

def txt_prnt(file, value, float_type = 0):
    # global vocab_txt
    last = 0
    for item in value:
        last += 1
        if float_type:
            item = format(item,'.10f') # it format the item explicitly to the float i.e 7.6325e-05, will formated as '0.00000763250' --type is str
        else:
            item = str(item)
        file.write(item)
        if last < len(value):
            file.write(" ")
        else:
            file.write("\n")


def svaingtotxt(path, file_no = 0):
    """
    This function is to save the herierchical tree and the initialization vectors
    to a text file
    :return:
    """
    x = file_no
    index2w_tmp = open(path+'\index2word'+str(x)+'.pkl', 'r')
    index2w = pkl.load(index2w_tmp)
    index2w_tmp.close()

    word2index = {}
    for ind in range(0,len(index2w)):
        tmp_val = index2w[ind]
        word2index[tmp_val] = ind



    vocab_tmp = open(path+'\\vocab'+str(x)+'.pkl','r')
    vocab = pkl.load(vocab_tmp)
    vocab_tmp.close()

    kmer_tmp = open(path+'\kmer'+str(x)+'.pkl','r')
    kmer1 = pkl.load(kmer_tmp)
    kmer_tmp.close()

    doctag_tmp = open(path+'\doctag'+str(x)+'.pkl','r')
    doctag = pkl.load(doctag_tmp)
    doctag_tmp.close()
    n_docs = doctag.shape[0]


    vocab_txt = open(path+'\\vocab_txt'+str(x)+'.txt', 'w')
    doctag_txt = open(path+'\doctag_txt'+str(x)+'.txt', 'w')

    """
    k = 0
    for kmer in sorted(vocab.keys()):
        vocab_txt.write(">" + kmer)
        vocab_txt.write("\n")
        comp_info = vocab[kmer]
        code = comp_info.code
        txt_prnt(vocab_txt, code)
        point = comp_info.point
        txt_prnt(vocab_txt, point)
        txt_prnt(vocab_txt, kmer1[k])
        print k
        k = k + 1
    """


    k = 0
    for kmer in sorted(vocab.keys()):
        vocab_txt.write(">" + kmer)
        vocab_txt.write("\n")
        comp_info = vocab[kmer]
        code = comp_info.code
        txt_prnt(vocab_txt, code)
        point = comp_info.point
        txt_prnt(vocab_txt, point)

        ind_kmer = word2index[kmer]
        txt_prnt(vocab_txt, kmer1[ind_kmer], float_type= 1)
        print k
        k = k + 1

    for doc in range(0, n_docs):
        doctag_txt.write(">" + str(doc))
        doctag_txt.write("\n")
        txt_prnt(doctag_txt, doctag[doc])

    doctag_txt.close()
    vocab_txt.close()

if __name__ == '__main__':
    save_path = r"../data/inpfile"

    saveinit = 1 # for initialization and herierchical softmax

    if saveinit:
        for i in range(0,1):
            for j in range(0,1):
                svaingtotxt(save_path, file_no = j)

"""
#X

#Vocab(code:array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=uint8), count:4, index:456, point:array([2590, 2589, 2586, 2580, 2569, 2546, 2526, 2461, 2327, 2087, 1652], dtype=uint32), sample_int:4294967296)

"""
