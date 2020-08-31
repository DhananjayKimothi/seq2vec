from collections import namedtuple
from collections import defaultdict
import logging
from six import iteritems, itervalues, string_types
from gensim import utils, matutils
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray

import heapq
logger = logging.getLogger(__name__)
from timeit import default_timer

from gensim.utils import keep_vocab_item

class DocvecsArray(utils.SaveLoad):
    """
    Default storage of doc vectors during/after training, in a numpy array.

    As the 'docvecs' property of a Doc2Vec model, allows access and
    comparison of document vectors.

    >>> docvec = d2v_model.docvecs[99]
    >>> docvec = d2v_model.docvecs['SENT_99']  # if string tag used in training
    >>> sims = d2v_model.docvecs.most_similar(99)
    >>> sims = d2v_model.docvecs.most_similar('SENT_99')
    >>> sims = d2v_model.docvecs.most_similar(docvec)

    If only plain int tags are presented during training, the dict (of
    string tag -> index) and list (of index -> string tag) stay empty,
    saving memory.

    Supplying a mapfile_path (as by initializing a Doc2Vec model with a
    'docvecs_mapfile' value) will use a pair of memory-mapped
    files as the array backing for doctag_syn0/doctag_syn0_lockf values.

    The Doc2Vec model automatically uses this class, but a future alternative
    implementation, based on another persistence mechanism like LMDB, LevelDB,
    or SQLite, should also be possible.
    """
    def __init__(self, mapfile_path=None):
        self.doctags = {}  # string -> Doctag (only filled if necessary)
        self.max_rawint = -1  # highest rawint-indexed doctag
        self.offset2doctag = []  # int offset-past-(max_rawint+1) -> String (only filled if necessary)
        self.count = 0
        self.mapfile_path = mapfile_path

    def note_doctag(self, key, document_no, document_length):
        """Note a document tag during initial corpus scan, for structure sizing."""
        if isinstance(key, int):
            self.max_rawint = max(self.max_rawint, key)
        else:
            if key in self.doctags:
                self.doctags[key] = self.doctags[key].repeat(document_length)
            else:
                self.doctags[key] = Doctag(len(self.offset2doctag), document_length, 1)
                self.offset2doctag.append(key)
        self.count = self.max_rawint + 1 + len(self.offset2doctag)

    def indexed_doctags(self, doctag_tokens):
        """Return indexes and backing-arrays used in training examples."""
        return ([self._int_index(index) for index in doctag_tokens if index in self],
                self.doctag_syn0, self.doctag_syn0_lockf, doctag_tokens)

    def trained_item(self, indexed_tuple):
        """Persist any changes made to the given indexes (matching tuple previously
        returned by indexed_doctags()); a no-op for this implementation"""
        pass

    def _int_index(self, index):
        """Return int index for either string or int index"""
        if isinstance(index, int):
            return index
        else:
            return self.max_rawint + 1 + self.doctags[index].offset

    def _key_index(self, i_index, missing=None):
        """Return string index for given int index, if available"""
        warnings.warn("use DocvecsArray.index_to_doctag", DeprecationWarning)
        return self.index_to_doctag(i_index)

    def index_to_doctag(self, i_index):
        """Return string key for given i_index, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - self.max_rawint - 1
        if 0 <= candidate_offset < len(self.offset2doctag):
            return self.offset2doctag[candidate_offset]
        else:
            return i_index

    def __getitem__(self, index):
        """
        Accept a single key (int or string tag) or list of keys as input.

        If a single string or int, return designated tag's vector
        representation, as a 1D numpy array.

        If a list, return designated tags' vector representations as a
        2D numpy array: #tags x #vector_size.
        """
        if isinstance(index, string_types + (int,)):
            return self.doctag_syn0[self._int_index(index)]

        return vstack([self[i] for i in index])

    def __len__(self):
        return self.count

    def __contains__(self, index):
        if isinstance(index, int):
            return index < self.count
        else:
            return index in self.doctags

    def borrow_from(self, other_docvecs):
        self.count = other_docvecs.count
        self.doctags = other_docvecs.doctags
        self.offset2doctag = other_docvecs.offset2doctag

    def clear_sims(self):
        self.doctag_syn0norm = None

    def estimated_lookup_memory(self):
        """Estimated memory for tag lookup; 0 if using pure int tags."""
        return 60 * len(self.offset2doctag) + 140 * len(self.doctags)

    def reset_weights(self, model):
        length = max(len(self.doctags), self.count)
        if self.mapfile_path:
            self.doctag_syn0 = np_memmap(self.mapfile_path+'.doctag_syn0', dtype=REAL,
                                         mode='w+', shape=(length, model.vector_size))
            self.doctag_syn0_lockf = np_memmap(self.mapfile_path+'.doctag_syn0_lockf', dtype=REAL,
                                               mode='w+', shape=(length,))
            self.doctag_syn0_lockf.fill(1.0)
        else:
            self.doctag_syn0 = empty((length, model.vector_size), dtype=REAL)
            self.doctag_syn0_lockf = ones((length,), dtype=REAL)  # zeros suppress learning

        for i in xrange(length):
            # construct deterministic seed from index AND model seed
            seed = "%d %s" % (model.seed, self.index_to_doctag(i))
            self.doctag_syn0[i] = model.seeded_vector(seed)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'doctag_syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of doc weight vectors")
            if replace:
                for i in xrange(self.doctag_syn0.shape[0]):
                    self.doctag_syn0[i, :] /= sqrt((self.doctag_syn0[i, :] ** 2).sum(-1))
                self.doctag_syn0norm = self.doctag_syn0
            else:
                if self.mapfile_path:
                    self.doctag_syn0norm = np_memmap(
                        self.mapfile_path+'.doctag_syn0norm', dtype=REAL,
                        mode='w+', shape=self.doctag_syn0.shape)
                else:
                    self.doctag_syn0norm = empty(self.doctag_syn0.shape, dtype=REAL)
                np_divide(self.doctag_syn0, sqrt((self.doctag_syn0 ** 2).sum(-1))[..., newaxis], self.doctag_syn0norm)

    def most_similar(self, positive=[], negative=[], topn=10, clip_start=0, clip_end=None):
        """
        Find the top-N most similar docvecs known from training. Positive docs contribute
        positively towards the similarity, negative docs negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given docs. Docs may be specified as vectors, integer indexes
        of trained docvecs, or if the documents were originally presented with string tags,
        by the corresponding tags.

        The 'clip_start' and 'clip_end' allow limiting results to a particular contiguous
        range of the underlying doctag_syn0norm vectors. (This may be useful if the ordering
        there was chosen to be significant, such as more popular tag IDs in lower indexes.)
        """
        self.init_sims()
        clip_end = clip_end or len(self.doctag_syn0norm)

        if isinstance(positive, string_types + integer_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each doc, if not already present; default to 1.0 for positive and -1.0 for negative docs
        positive = [
            (doc, 1.0) if isinstance(doc, string_types + (ndarray,) + integer_types)
            else doc for doc in positive
        ]
        negative = [
            (doc, -1.0) if isinstance(doc, string_types + (ndarray,) + integer_types)
            else doc for doc in negative
        ]

        # compute the weighted average of all docs
        all_docs, mean = set(), []
        for doc, weight in positive + negative:
            if isinstance(doc, ndarray):
                mean.append(weight * doc)
            elif doc in self.doctags or doc < self.count:
                mean.append(weight * self.doctag_syn0norm[self._int_index(doc)])
                all_docs.add(self._int_index(doc))
            else:
                raise KeyError("doc '%s' not in trained set" % doc)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.doctag_syn0norm[clip_start:clip_end], mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_docs), reverse=True)
        # ignore (don't return) docs from the input
        result = [(self.index_to_doctag(sim), float(dists[sim])) for sim in best if sim not in all_docs]
        return result[:topn]

    def doesnt_match(self, docs):
        """
        Which doc from the given list doesn't go with the others?

        (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        """
        self.init_sims()

        docs = [doc for doc in docs if doc in self.doctags or 0 <= doc < self.count]  # filter out unknowns
        logger.debug("using docs %s" % docs)
        if not docs:
            raise ValueError("cannot select a doc from an empty list")
        vectors = vstack(self.doctag_syn0norm[self._int_index(doc)] for doc in docs).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, docs))[0][1]

    def similarity(self, d1, d2):
        """
        Compute cosine similarity between two docvecs in the trained set, specified by int index or
        string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))

    def n_similarity(self, ds1, ds2):
        """
        Compute cosine similarity between two sets of docvecs from the trained set, specified by int
        index or string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        """
        v1 = [self[doc] for doc in ds1]
        v2 = [self[doc] for doc in ds2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

class Doctag(namedtuple('Doctag', 'offset, word_count, doc_count')):
    """A string document tag discovered during the initial vocabulary
    scan. (The document-vector equivalent of a Vocab object.)

    Will not be used if all presented document tags are ints.

    The offset is only the true index into the doctags_syn0/doctags_syn0_lockf
    if-and-only-if no raw-int tags were used. If any raw-int tags were used,
    string Doctag vectors begin at index (max_rawint + 1), so the true index is
    (rawint_index + 1 + offset). See also DocvecsArray.index_to_doctag().
    """
    __slots__ = ()

    def repeat(self, word_count):
        return self._replace(word_count=self.word_count + word_count, doc_count=self.doc_count + 1)
