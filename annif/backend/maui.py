"""Maui backend that makes calls to a Maui Server instance using its API"""


import time
import os.path
import json
import requests
import requests.exceptions
from annif.exception import NotSupportedException, OperationFailedException
from annif.suggestion import SubjectSuggestion, ListSuggestionResult
from . import backend


class MauiBackend(backend.AnnifBackend):
    name = "maui"

    TRAIN_FILE = 'maui-train.jsonl'

    @property
    def endpoint(self):
        return self.params['endpoint']

    @property
    def tagger(self):
        return self.params['tagger']

    @property
    def tagger_url(self):
        return self.params['endpoint'] + self.params['tagger']

    def _initialize_tagger(self):
        self.info("Initializing Maui Service tagger '{}'".format(self.tagger))

        # try to delete the tagger in case it already exists
        resp = requests.delete(self.tagger_url)
        self.debug("Trying to delete tagger {} returned status code {}"
                   .format(self.tagger, resp.status_code))

        # create a new tagger
        data = {'id': self.tagger, 'lang': self.params['language']}
        try:
            resp = requests.post(self.endpoint, data=data)
            self.debug("Trying to create tagger {} returned status code {}"
                       .format(self.tagger, resp.status_code))
            resp.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise OperationFailedException(err)

    def _upload_vocabulary(self, project):
        self.info("Uploading vocabulary")
        try:
            resp = requests.put(self.tagger_url + '/vocab',
                                data=project.vocab.as_skos())
            resp.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise OperationFailedException(err)

    def _create_train_file(self, corpus):
        self.info("Creating train file")
        train_path = os.path.join(self.datadir, self.TRAIN_FILE)
        with open(train_path, 'w') as train_file:
            for doc in corpus.documents:
                doc_obj = {'content': doc.text, 'topics': list(doc.labels)}
                json_doc = json.dumps(doc_obj)
                print(json_doc, file=train_file)

    def _upload_train_file(self):
        self.info("Uploading training documents")
        train_path = os.path.join(self.datadir, self.TRAIN_FILE)
        with open(train_path, 'rb') as train_file:
            try:
                resp = requests.post(self.tagger_url + '/train',
                                     data=train_file)
                resp.raise_for_status()
            except requests.exceptions.RequestException as err:
                raise OperationFailedException(err)

    def _wait_for_train(self):
        self.info("Waiting for training to be completed...")
        while True:
            try:
                resp = requests.get(self.tagger_url + "/train")
                resp.raise_for_status()
            except requests.exceptions.RequestException as err:
                raise OperationFailedException(err)

            response = resp.json()
            if response['completed']:
                self.info("Training completed.")
                return
            time.sleep(1)

    def train(self, corpus, project):
        if corpus.is_empty():
            raise NotSupportedException('training backend {} with no documents'
                                        .format(self.backend_id))
        self._initialize_tagger()
        self._upload_vocabulary(project)
        self._create_train_file(corpus)
        self._upload_train_file()
        self._wait_for_train()

    def _suggest(self, text, project, params):
        data = {'text': text}

        try:
            resp = requests.post(self.tagger_url + '/suggest', data=data)
            resp.raise_for_status()
        except requests.exceptions.RequestException as err:
            self.warning("HTTP request failed: {}".format(err))
            return ListSuggestionResult([], project.subjects)

        try:
            response = resp.json()
        except ValueError as err:
            self.warning("JSON decode failed: {}".format(err))
            return ListSuggestionResult([], project.subjects)

        try:
            return ListSuggestionResult(
                [SubjectSuggestion(uri=h['id'],
                                   label=h['label'],
                                   score=h['probability'])
                 for h in response['topics']
                 if h['probability'] > 0.0], project.subjects)
        except (TypeError, ValueError) as err:
            self.warning("Problem interpreting JSON data: {}".format(err))
            return ListSuggestionResult([], project.subjects)
