# seq2vec: Learning Distributed representations for molecular sequences


### Step by Step process
1. Put fasta file in **data**
2. Run (from **gen_initialization_files_for_c_code folder**)
	* s2v_c.py
        * set *variables* : file
    * vocab_text.py

   outputs will be saved in data/inpfiles/

2. use seq2vec_train for training the model
    * set *variables* : path_save_model, path_file_to_be_processed, path_save_vectors, doctag_path, vocab_path

    * outputs will be saved in data/model/

3. use seq2vec_ppi_infr for generating the sequence embeddings (.binary)
    * set *variables* : path_model, path_file_to_be_processed, path_save_vectors

    * output will be saved in data/embeddings/

4. Once the vectors are generated use read_binary.py to pickle them as dictionary

    * output will be saved in data/embeddings/
    * set *variables* : path, fileIn, fileOut

5. vectors can be accessed using key = 'sequence_vectors'

## Example:
* yeast sequence file (yeast_seq.fasta) is given in /data/
* the corresponding embeddings and all other files are provided in /data/embeddings/
* to verify -- delete all the files in data/inpfiles; data/model; data/embeddings and run the files following the steps mentioned above. Default parameter settings will generate the deleted files.

**If using this module please cite**

@article{kimothi2016distributed,
  title={Distributed representations for biological sequence analysis},
  author={Kimothi, Dhananjay and Soni, Akshay and Biyani, Pravesh and Hogan, James M},
  journal={arXiv preprint arXiv:1608.05949},
  year={2016}
}
