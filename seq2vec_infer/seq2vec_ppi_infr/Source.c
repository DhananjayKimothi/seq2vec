//  non overlapping window for  training
// overlapping window for testing
// hardcoded -- The path of input fasta file, sequence vec, kmer vec and vocab
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h> // computation time

#define MaxSequences 500000 // ?
#define K 3 // kmer size
#define MAXLINE 2000  //? it was 128
#define SigLength 100
#define Window 1 // context size
#define min_alpha 0.005000 // step size

int file_n;

clock_t start, end1, end2;
double training_time;
double inference_time;


typedef float real;


int* lengths;
int alphabet_size;


int num_sequences, num_kmers, num_families, num_nodes; // counts
char*sequences[MaxSequences]; // ?
int family_name[10000]; // 10
int family_size[10000]; // samples in a family
int family_start[10000]; // ?
int alphabet_size; // ?
int depth; // depth of the binary tree (herierchical softmax)
int last_family; // variable to store last family


int batch_words = 1000; // kmers in a batch after which step size is changed
int MaxNode = 0, MinNode = 1000000;  // variables for stor
int point_size = 100;

double** seq_vector, **kmer_vector, **vocab_vector; // store vectors
int** codes, **points; // To store the binary code and the path for herierchical tree

int reverse[256]; // mapping kmer to dictionary index
char* path_model;
char* path_file_to_be_processed;
char* path_save_vectors;


//path for files generated from python -- vector initialization and the vocab tree


void ReadFASTA(char *filename)
{

	printf("\treading %s...\n", filename);
	num_families = 0;
	num_sequences = 0;
	last_family = -1;
	char  line[MAXLINE];
	FILE *fp = fopen(filename, "r");
	fgets(line, MAXLINE, fp);


	while (1)
	{
		if (line[0] != '>')
			break;

		//int family = atoi(strrchr(line, '_') + 1);

		char* seq_start = NULL;
		int seq_length = 0;

		while (fgets(line, MAXLINE, fp))
		{
			if (line[0] == '>')
				break;

			int line_length = strlen(line) - 1;
			line[line_length] = '\0';

			seq_start = realloc(seq_start, seq_length + line_length + 1);
			strcpy(seq_start + seq_length, line);
			seq_length += line_length;
		}
		sequences[num_sequences++] = seq_start;

	}
	fclose(fp);
}

int RandomInt(int max)
{
	return max * ((double)rand() / RAND_MAX);
}
double RandomDouble()
{
	return ((double)rand() / RAND_MAX) - 0.5;
}

void SaveSequences(char* path)
{
	char filename[256];
	sprintf(filename, "%s", path, file_n);

	printf("\twriting %s...\n", filename);
	FILE* file = fopen(filename, "wb");

	fwrite(&num_families, sizeof(int), 1, file);
	fwrite(&family_name, sizeof(int), num_families, file);
	fwrite(&family_size, sizeof(int), num_families, file);

	fwrite(&num_sequences, sizeof(int), 1, file);
	for (int i = 0; i < num_sequences; i++)
		fwrite(seq_vector[i], sizeof(double), SigLength, file);

	fclose(file);
}

void LoadModel(char* path)
{
	char filename[256];
	sprintf(filename, "%s", path);
	printf("\treading %s...\n", filename);
	FILE* file = fopen(filename, "rb");

	fread(&num_kmers, sizeof(int), 1, file);
	fread(&num_nodes, sizeof(int), 1, file);
	lengths = (int*)malloc(sizeof(int) * num_kmers);
	fread(lengths, sizeof(int), num_kmers, file);

	kmer_vector = (double**)malloc(sizeof(double*)*num_kmers);
	for (int i = 0; i < num_kmers; i++)
	{
		kmer_vector[i] = (double*)malloc(sizeof(double)*SigLength);
		fread(kmer_vector[i], sizeof(double), SigLength, file);
		/*
		for (int j = 0; j = 3; j++)
		{

		printf("%f\n", kmer_vector[i][j]);
		}*/

	}

	vocab_vector = (double**)malloc(sizeof(double*) * num_nodes);
	for (int i = 0; i < num_nodes; i++)
	{
		vocab_vector[i] = (double*)malloc(sizeof(double)*SigLength);
		fread(vocab_vector[i], sizeof(double), SigLength, file);
	}


	codes = (int**)malloc(sizeof(int*)*num_kmers);

	for (int i = 0; i < num_kmers; i++)
	{
		codes[i] = (int*)malloc(sizeof(int)*lengths[i]);
		fread(codes[i], sizeof(int), lengths[i], file);
	}


	points = (int**)malloc(sizeof(int*)*num_kmers);
	for (int i = 0; i < num_kmers; i++)
	{
		points[i] = (int*)malloc(sizeof(int)*lengths[i]);
		fread(points[i], sizeof(int), lengths[i], file);

	}


	/*for (int i = 0; i < 20; i++)
	{
	printf("%d ", lengths[i]);
	}*/



	seq_vector = (double**)malloc(sizeof(double*) * num_sequences);
	for (int j = 0; j < num_sequences; j++)
	{
		seq_vector[j] = (double*)malloc(sizeof(double) * SigLength);
		for (int i = 0; i < SigLength; i++)
			seq_vector[j][i] = 0;
		//RandomDouble() / SigLength;
	}

	fclose(file);
}

int KmerIndex(char* kmer)
{
	int index = 0;
	for (int i = 0; i < K; i++)
		index = index * alphabet_size + reverse[kmer[i]];
	return index;
}

// testNum: # splits; Training : 1 for training the nueral network, 0 for inference stage
void Process()
{

	char fastaFilename[256]; // ?
	sprintf(fastaFilename, "%s", path_file_to_be_processed);


	int Iterations = 1;
	int Steps = 5;


	char* alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"; // 25 alphabets --- not so sure alphabets are also considered
												  //char* alphabet = "ACDEFGHIKLMNPQRSTUVWY";

	srand(42); // ? probably its like seed
	alphabet_size = strlen(alphabet); // string size : 25 in this case
	for (int i = 0; i < alphabet_size; i++)
		reverse[alphabet[i]] = i;

	ReadFASTA(fastaFilename); // reading fasta file .. see function

	printf("\t%s: %d Families, %d Sequences\n", fastaFilename, num_families, num_sequences);


	double next_alpha; // step size
	double next_alpha1;
	double alpha;
	int pushed_words = 0; // to track the progress the alpha depends on the progress
	int total_words = 0; // total number of words
	int total_pushed_words = 0;
	int pushed_examples = 0;
	int total_examples = num_sequences * Iterations;
	int document_based = 1;
	int word_based = 0;
	// counting total words for the number of iterations

	for (int seq_idx = 0; seq_idx < num_sequences; seq_idx++)
		total_words += strlen(sequences[seq_idx]);
	alpha = 0.025;
	total_words *= Iterations;
	next_alpha1 = alpha;
	double min_alphai = 0.0001;
	LoadModel(path_model);

	// Algorithm

	double neu2e_inlier[SigLength];
	double neu2e_outlier[SigLength];
	double neu2e[SigLength];
	double neu1e[SigLength];
	double Sig_seq_inlier = 0;
	int batch_size = 0;
	next_alpha = next_alpha1;
	int seq_idx_tst = 79;
	//num_sequences
	for (int seq_idx = 0; seq_idx < num_sequences; seq_idx++) // iteration over multiple epochs
	{
		printf("(%d, %d ) ", seq_idx, num_sequences-seq_idx);
		//printf("iteration no. %d", iterations + 1);
		int seq_len = 0; // kmers in the sequence
		double alphai = 0.1; // for each sequence
		char* seq = sequences[seq_idx];  // seq is a pointer which is pointing to the starting location of sequence

										 // Non Overlap
		for (int step = 0; step < Steps; step++)
		{

			next_alpha = alphai;

			// for non overlapping
			for (int slide = 0; slide < K; slide++)
			{
				pushed_words += floor((strlen(seq) - slide) / K);
				total_pushed_words += floor((strlen(seq) - slide) / K);
				int pos = 0;
				for (pos = pos + slide; pos < strlen(seq) - K + 1; pos = pos + K) // iteration over a squence
				{
					char* kmer = seq + pos; // kmer is a pointing to the starting location of "to be predicted kmer"


					double l1[SigLength]; // input signal
					for (int i = 0; i < SigLength; i++)
					{
						//assert(seq_vector[seq_idx][i] <= 1.5);
						l1[i] = seq_vector[seq_idx][i];
					}

					int context[Window * 2]; //
					int context_size = 0; // temp variable
					int reduced_window = RandomInt(Window - 1);
					//int reduced_window = 0;// fix context
					for (int offset = K; offset <= (Window*K) - reduced_window; offset = offset + K) // predicting kmer given its context
					{
						if (pos + offset < strlen(seq) - K + 1)
						{
							char* kmer2 = kmer + offset; // collecting kmers from right
							int kmer2_idx = KmerIndex(kmer2);
							context[context_size++] = kmer2_idx; // storing the indices in the array (is it array?)
							for (int i = 0; i < SigLength; i++)
							{
								l1[i] += kmer_vector[kmer2_idx][i];
							}

						}
						if (0 <= pos - offset)
						{
							char* kmer2 = kmer - offset; // collecting kmers from the left
							int kmer2_idx = KmerIndex(kmer2);
							context[context_size++] = kmer2_idx;
							for (int i = 0; i < SigLength; i++)
							{
								l1[i] += kmer_vector[kmer2_idx][i];
							}
						}
					}

					/*
					// for overlapping
					for (int pos = 0; pos < strlen(seq) - K + 1; pos++) // iteration over a squence
					{
					char* kmer = seq + pos; // kmer is a pointing to the starting location of "to be predicted kmer"
					double l1[SigLength]; // input signal
					for (int i = 0; i < SigLength; i++)
					{
					l1[i] = seq_vector[seq_idx][i];
					}
					int context[Window * 2]; //
					int context_size = 0; // temp variable
					int reduced_window = RandomInt(Window - 1);
					for (int offset = 1; offset <= Window - reduced_window; offset++) // predicting kmer given its context
					{
					if (pos + offset < strlen(seq) - K + 1)
					{
					char* kmer2 = kmer + offset; // collecting kmers from right
					int kmer2_idx = KmerIndex(kmer2);
					context[context_size++] = kmer2_idx; // storing the indices in the array (is it array?)
					for (int i = 0; i < SigLength; i++)
					{
					l1[i] += kmer_vector[kmer2_idx][i];
					}

					}
					if (0 <= pos - offset)
					{
					char* kmer2 = kmer - offset; // collecting kmers from the left
					int kmer2_idx = KmerIndex(kmer2);
					context[context_size++] = kmer2_idx;
					for (int i = 0; i < SigLength; i++)
					l1[i] += kmer_vector[kmer2_idx][i];
					}

					}*/


					// Code below is same for choice of context, overlapping or non overlapping.
					// Imp
					// We are using average of the context for feeding in the NN, addition or concatenation are other two important aspects of it.

					int count = 1 + context_size;
					if (count > 1)
						for (int i = 0; i < SigLength; i++)
							l1[i] /= count;
					// Train CBOW ..
					int kmer_index = KmerIndex(kmer); //index of the kmer
					memset(neu1e, 0, sizeof(neu1e)); // initializing with zeros
					int* path = points[kmer_index]; // path is a pointer to the
					depth = lengths[kmer_index];
					for (int h = 0; h < depth; h++)
					{
						int node_idx = path[h];
						double dot = 0;
						for (int i = 0; i < SigLength; i++)
							dot += vocab_vector[node_idx][i] * l1[i];
						double fa = 1.0 / (1.0 + exp(-dot));
						double ga = (1.0 - codes[kmer_index][h] - fa) * next_alpha;
						for (int i = 0; i < SigLength; i++)
							neu1e[i] += ga * vocab_vector[node_idx][i];
					}
					for (int i = 0; i < SigLength; i++)
						seq_vector[seq_idx][i] += neu1e[i];

					//printf("(%d, 0) %f", seq_idx, seq_vector[seq_idx][0]);
				}
			}
			alphai = ((alphai - min_alphai) / (Steps - step)) + min_alphai;
		}


	}

	SaveSequences(path_save_vectors);


	for (int i = 0; i < num_kmers; i++)
	{
		free(codes[i]);
		free(points[i]);
	}

	for (int j = 0; j < num_nodes; j++)
		free(vocab_vector[j]);

	for (int kmer = 0; kmer < num_kmers; kmer++)
		free(kmer_vector[kmer]);

	for (int j = 0; j < num_sequences; j++)
		free(seq_vector[j]);

	free(vocab_vector);
	free(kmer_vector);
	free(seq_vector);
	free(codes);
	free(points);
}

int main(int argc, char *argv[])
{
	/*
	path_model = argv[1];
	printf("%s\n", path_model);
	path_file_to_be_processed = argv[2];
	printf("%s\n", path_file_to_be_processed);
	path_save_vectors = argv[3];
	printf("%s\n", path_save_vectors);
	*/
	path_model = "..\\..\\data\\model\\M_s2v.binary";
	path_file_to_be_processed = "..\\..\\data\\yeast_seq.fasta";
	path_save_vectors = "..\\..\\data\\embeddings\\yeast_s2v.binary";
	
	start = clock();

	Process();		  //Process(test, 0); // testing
	end1 = clock();

	inference_time = ((double)(end1 - start)) / CLOCKS_PER_SEC;
	printf("Inference time is %f sec ", inference_time);


	int a;

	printf("Please input an integer value: ");
	scanf("%d", &a);


}
