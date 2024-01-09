"""Annif backend using a gensim embedding model"""
from __future__ import annotations

import os.path

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import rrflow
import rrflow.utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import typing
from collections import defaultdict
import math
import joblib

from annif.exception import NotInitializedException, NotSupportedException
from annif.suggestion import SubjectSuggestion

from . import backend, mixins

from annif.corpus.document import DocumentCorpus
from collections.abc import Iterator


class GensimBackend(backend.AnnifBackend):
    """gensim backend for Annif using Doc2Vec embeddings"""

    name = "gensim"

    MODEL_FILE = "gensim-model"

    # defaults for uninitialized instances
    _doc2vec = None
    _idf_dict = defaultdict(float)
    _subject_dict = defaultdict(list)
    _model = None

    def default_params(self) -> dict[str, Any]:
        params = backend.AnnifBackend.DEFAULT_PARAMETERS.copy()
        params.update(mixins.ChunkingBackend.DEFAULT_PARAMETERS)
        params.update(self.DEFAULT_PARAMETERS)
        return params

    def initialize(self, parallel: bool = False) -> None:
        if self._doc2vec is None:
            path = os.path.join(self.datadir, self.MODEL_FILE)
            self.debug("loading gensim model from {}".format(path))
            if os.path.exists(path):
                self._model = joblib.load(path)
                self._doc2vec = self._model['doc2vec']
                self._idf_dict = self._model['idf_dict']
                self._subject_dict = self._model['subject_dict']
                self.debug("loaded model from {}".format(path))
            else:
                raise NotInitializedException(
                    "model {} not found".format(path), backend_id=self.backend_id
                )

    def _normalize_text(self, text: str) -> str:
        return self.project.analyzer.tokenize_words(text)

    def _create_model(self, doc_corpus: DocumentCorpus) -> None:
        self.info("creating gensim Doc2Vec model")
        modelpath = os.path.join(self.datadir, self.MODEL_FILE)
        self._create_doc2vec_and_dicts(doc_corpus=doc_corpus)
        joblib.dump(self._model, modelpath)

    def _create_doc2vec_and_dicts(self, doc_corpus: DocumentCorpus) -> None:
        corpus = {}
        doc_freq = defaultdict(int)
        article_count = 0
        for doc_index, d in enumerate(doc_corpus.documents):
            content = d.text
            clean_content = rrflow.utils.clean_article_content(content)
            words = self._normalize_text(clean_content)
            corpus[doc_index] = TaggedDocument(words, tags=[doc_index])
            self._subject_dict[doc_index] = d.subject_set._subject_ids
            for subject_id in d.subject_set._subject_ids:
                doc_freq[subject_id] += 1
            article_count = doc_index
        self._idf_dict = {s:math.log(article_count/(1 + doc_freq[s])) for s in doc_freq.keys()}
        self._doc2vec = Doc2Vec(corpus.values())
        self._model = {
            'doc2vec': self._doc2vec,
            'idf_dict': self._idf_dict,
            'subject_dict': self._subject_dict
        }


    def _train(
        self,
        corpus: DocumentCorpus,
        params: dict[str, Any],
        jobs: int = 0,
    ) -> None:
        if corpus != "cached":
            if corpus.is_empty():
                raise NotSupportedException(
                    "training backend {} with no documents".format(self.backend_id)
                )
        else:
            self.info("Reusing cached training data from previous run.")
        self._create_model(corpus)


    def get_article_nearest_neighbors(self, article_text: str, limit:int=10, doc2vec_model: Doc2Vec=None) -> typing.List[typing.Tuple[int,float]]:
        """
        Returns the embedding of the given article.
        """
        art_clean_content = rrflow.utils.clean_article_content(article_text)
        words = self._normalize_text(art_clean_content)
        embedding = doc2vec_model.infer_vector(words)
        return doc2vec_model.dv.most_similar([embedding], topn=limit)

    def get_article_subjects(self, article_text: str, limit:int=5, doc2vec_model:Doc2Vec=None) -> typing.Dict[str, float]:
        """
        Returns a dict of subjects and their weights.
        """
        nearest_neighbors = self.get_article_nearest_neighbors(article_text, limit, doc2vec_model)
        subjects_dict = defaultdict(float)
        for article_id, distance in nearest_neighbors:
            article_subject_ids = self._subject_dict[article_id]
            for a_subj_index in article_subject_ids: # for now, we just do a simple count, not tdf-idf
#                subject_text = subject_index._subjects[a_subj_index].labels['da']
#                tf = (min(article_text.count(subject_text),1) / len(article_text)) if len(article_text) > 0 else 0
#                idf = idf_dict[a_subj_index]
#                subjects_dict[a_subj_index] += (1 / distance) * (tf / idf)
                subjects_dict[a_subj_index] += (1 / distance)
        res = dict(subjects_dict)
        res = [(s, res[s]) for s in sorted(res, key=res.get, reverse=True)[:limit]]
        # normalize, so that values sum up to 1
        norm_val = 1 / sum([v for _, v in res])
        return {s: v * norm_val for s, v in res}


    def _suggest(self, article_text: str, params: dict[str, Any]) -> Iterator:
        self.debug(
            'Suggesting subjects for text "{}..." (len={})'.format(article_text[:20], len(article_text))
        )
        suggestion_dict = self.get_article_subjects(article_text, int(params["limit"]), self._doc2vec)
        for s, v in suggestion_dict.items():
            yield SubjectSuggestion(subject_id=s, score=v)
