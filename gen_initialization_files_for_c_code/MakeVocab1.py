#!/usr/bin/env python
# -*- coding: utf-8 -*-c
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
from collections import defaultdict
import logging
from six import iteritems, itervalues, string_types
from gensim import utils, matutils
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL, \
    uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis, \
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray

import heapq

#logger = logging.getLogger(__name__)
from timeit import default_timer

from gensim.utils import keep_vocab_item
from DocVecsArray import DocvecsArray

class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class BuildVocab(object):
    def __init__(self):
        self.min_count = 1
        self.sample = 0
        self.max_vocab_size = 10000
        self.docvecs = DocvecsArray()
        self.hs = 1
        self.vector_size = 100  # check this --its a vector size
        self.layer1_size = self.vector_size
        self.negative = 0
        self.sorted_vocab = 0  # want to sort vocab on the basis of frequency ?
        self.null_word = 0  # check this
        self.dm = 1
        self.dm_concat = 0
        self.seed = 1
        self.hashfxn = hash
        self.total_words = 0

    def scan_vocab(self, documents, progress_per=10000, trim_rule=None):
        #logger.info("collecting all words and their counts")
        document_no = -1
        total_words = 0
        min_reduce = 1
        interval_start = default_timer() - 0.00001  # guard against next sample being identical
        interval_count = 0
        vocab = defaultdict(int)
        for document_no, document in enumerate(documents):

            if document_no % progress_per == 0:
                interval_rate = (total_words - interval_count) / (default_timer() - interval_start)
                #logger.info("PROGRESS: at example #%i, processed %i words (%i/s), %i word types, %i tags",
                 #           document_no, total_words, interval_rate, len(vocab), len(self.docvecs))
                interval_start = default_timer()
                interval_count = total_words
            document_length = len(document.words)

            for tag in document.tags:
                self.docvecs.note_doctag(tag, document_no, document_length)

            for word in document.words:
                vocab[word] += 1
            total_words += len(document.words)

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        #logger.info("collected %i word types and %i unique tags from a corpus of %i examples and %i words",
         #           len(vocab), len(self.docvecs), document_no + 1, total_words)
        self.corpus_count = document_no + 1
        self.raw_vocab = vocab
        self.total_words = total_words  # dk

    def scale_vocab(self, min_count=None, sample=None, dry_run=False, keep_raw_vocab=False, trim_rule=None):
        """
        Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).

        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.

        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.

        """
        min_count = min_count or self.min_count
        sample = sample or self.sample

        # Discard words less-frequent than min_count
        if not dry_run:
            self.index2word = []
            # make stored settings match these applied settings
            self.min_count = min_count
            self.sample = sample
            self.vocab = {}
        drop_unique, drop_total, retain_total, original_total = 0, 0, 0, 0
        retain_words = []
        for word, v in iteritems(self.raw_vocab):
            if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                retain_words.append(word)
                retain_total += v
                original_total += v
                if not dry_run:
                    self.vocab[word] = Vocab(count=v, index=len(self.index2word))
                    self.index2word.append(word)
            else:
                drop_unique += 1
                drop_total += v
                original_total += v
        #logger.info("min_count=%d retains %i unique words (drops %i)",
         #           min_count, len(retain_words), drop_unique)
        #logger.info("min_count leaves %i word corpus (%i%% of original %i)",
         #           retain_total, retain_total * 100 / max(original_total, 1), original_total)

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.vocab[w].sample_int = int(round(word_probability * 2 ** 32))

        if not dry_run and not keep_raw_vocab:
            #logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        #logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        #logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
         #           downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total)

        # return from each step: words-affected, resulting-corpus-size
        report_values = {'drop_unique': drop_unique, 'retain_total': retain_total,
                         'downsample_unique': downsample_unique, 'downsample_total': int(downsample_total)}

        # print extra memory estimates
        report_values['memory'] = self.estimate_memory(vocab_size=len(retain_words))

        return report_values

    def reset_weightsW2vec(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        #logger.info("resetting layer weights")
        self.syn0 = empty((len(self.vocab), self.vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            # construct deterministic seed from word AND seed argument
            self.syn0[i] = self.seeded_vector(self.index2word[i] + str(self.seed))
        if self.hs:
            self.syn1 = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        self.syn0norm = None

        self.syn0_lockf = ones(len(self.vocab), dtype=REAL)  # zeros suppress learning

    def reset_weights(self):
        if self.dm and self.dm_concat:
            # expand l1 size to match concatenated tags+words length
            self.layer1_size = (self.dm_tag_count + (2 * self.window)) * self.vector_size
            #logger.info("using concatenative %d-dimensional layer1" % (self.layer1_size))
        self.reset_weightsW2vec()
        self.docvecs.reset_weights(self)

    def seeded_vector(self, seed_string):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        return (once.rand(self.vector_size) - 0.5) / self.vector_size

    def finalize_vocab(self):
        """Build tables and model weights based on final vocabulary settings."""
        if not self.index2word:
            self.scale_vocab()
        if self.sorted_vocab:
            self.sort_vocab()
        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(self.vocab)
            self.index2word.append(word)
            self.vocab[word] = v
        # set initial input/projection and hidden weights
        self.reset_weights()

    def sort_vocab(self):
        """Sort the vocabulary so that most frequent words have the lowest indexes"""
        if hasattr(self, 'syn0'):
            raise RuntimeError("must sort before initializing vectors/weights")
        self.index2word.sort(key=lambda word: self.vocab[word].count, reverse=True)
        #self.index2word.sort()
        for i, word in enumerate(self.index2word):
            self.vocab[word].index = i

    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.
        """
        #logger.info("constructing a huffman tree from %i words", len(self.vocab))

        # build the huffman tree
        heap = list(itervalues(self.vocab))
        heapq.heapify(heap)
        for i in xrange(len(self.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(self.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(self.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            #logger.info("built huffman tree with maximum node depth %i", max_depth)


    def make_cum_table(self,power = 0.75, domain = 2**31-1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.
        To draw a word index, choose a random integer upto the maximum value in the table(cum_table[-1])
        then finding that integer's stored insertion point (as if by bisect_left or ndarray.searchsorted())
        That insertion point is the draw index, coming up in proportion equal to the increment at that slot.
        called internally from build.vocab()
        """

        vocab_size = len(self.index2word)
        self.cum_table = zeros(vocab_size, dtype = uint32)
        # compute sum of all power(Z in paper)

        train_words_pow = float(sum([self.vocab[word].count**power for word in self.vocab]))
        cumulative = 0.0
        for word_index in range(vocab_size):
            cumulative += self.vocab[self.index2word[word_index]].count**power /train_words_pow
            self.cum_table[word_index] = round(cumulative*domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain


    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.scan_vocab(sentences, trim_rule=trim_rule)  # initial survey
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab,
                         trim_rule=trim_rule)  # trim by min_count & precalculate downsampling
        self.finalize_vocab()

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings and provided vocabulary size."""
        vocab_size = vocab_size or len(self.vocab)
        report = report or {}
        report['vocab'] = vocab_size * (700 if self.hs else 500)
        report['syn0'] = vocab_size * self.vector_size * dtype(REAL).itemsize
        if self.hs:
            report['syn1'] = vocab_size * self.layer1_size * dtype(REAL).itemsize
        if self.negative:
            report['syn1neg'] = vocab_size * self.layer1_size * dtype(REAL).itemsize
        report['total'] = sum(report.values())
        #logger.info("estimated required memory for %i words and %i dimensions: %i bytes",
         #           vocab_size, self.vector_size, report['total'])
        return report
