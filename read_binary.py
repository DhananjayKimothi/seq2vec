
# coding: utf-8

# This script is same as read_biniary, the only difference is the input

#coding utf-8
from struct import *
import sys
import os
def main(Dir_path, fileIn, fileOut):
    import os
    # Dir_path = "E:\SeqReterivalWork\Top25"
    input_file = os.path.join(Dir_path, fileIn +'.binary')
    output_file = os.path.join(Dir_path, fileOut +'.pkl')
    v_s = 100 # vector size

    f = open(input_file,'rb')


    int_size = 4 #byte
    double_size = 8 #byte

    # num_families = 1
    # family_name = 25
    # family_size = 25
    # num_sequences = 1

    Database = {}
    family_name = []
    family_size = []
    sequences_vecs = []
    with open(input_file, 'rb') as f:


        # reading number of families:
        z = f.read(int_size)
        num_families = unpack('i',z)[0]

        # reading family names
        for i in list(range(1,num_families+1,1)):
            z = f.read(int_size)
            family_name.append(unpack('i',z)[0])


        # reading family_size
        for i in list(range(num_families+1,((num_families+1)*2)-1,1)):
            z = f.read(int_size)
            family_size.append(unpack('i',z)[0])


        # reading total number of sequences
        z = f.read(int_size)
        num_sequences = unpack('i',z)[0]
        print(num_sequences)

        for i in range(0,num_sequences,1):
            sequence_vec = []
            for  j in list(range(0,v_s,1)):
                z = f.read(double_size)
                sequence_vec.append(unpack('d',z)[0])

            sequences_vecs.append(sequence_vec)
    Database['num_families'] = num_families
    Database['family_names'] = family_name
    Database['family_sizes'] = family_size
    Database['sequence_vectors'] = sequences_vecs
    Database['info'] = """
    This dictionary contains: num_families, family_names -- numbering for families
    starting from 1, family_size and sequence_vectors - the sequence vectors are stored as a
    a list of lists. Sequences corresponding to a family can be accessed utilizing the family_sizes.
    """

    import pickle as pkl
    DB = open(output_file,'wb')
    pkl.dump(Database,DB)
    DB.close()


if __name__ == '__main__':
    path = "data/embeddings"
    fileIn = "yeast_s2v"
    fileOut = "yeast_s2v_embeddings"
    # print(path,file)
    main(path, fileIn,fileOut)







# In[189]:

"""
import pickle as pkl
import os
cd_path = os.getcwd()
file_path = os.path.join(cd_path, 'Database1.pkl')
in_file = open(file_path,'rb')
Database = pkl.load(in_file)
in_file.close()
print(Database['info'])
    # file = sys.argv[1]
    # path, tail = os.path.split(file)
    # tail, temp = tail.split('.')
	#print(file_no)


"""

#pkl.load()
