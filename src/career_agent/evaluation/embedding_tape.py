"""Exact-text embedding recordings for strict, service-free evaluation replay."""
from __future__ import annotations

import hashlib
import json


class MissingEmbeddingRecording(ValueError):
    pass


class EmbeddingTape:
    def __init__(self, client=None, *, recording=None):
        if (client is None) == (recording is None):
            raise ValueError("provide exactly one of client or recording")
        self.client = client
        self.model_id = client.model_id if client is not None else recording['model_id']
        self.records = {} if recording is None else dict(recording['records'])
        self.missing = set()

    def embed(self, texts):
        texts = list(texts)
        key = hashlib.sha256(json.dumps([self.model_id, texts], ensure_ascii=False).encode()).hexdigest()
        if key not in self.records:
            if self.client is None:
                self.missing.add(key)
                raise MissingEmbeddingRecording("embedding text/model absent from frozen recording")
            try:
                vectors = [list(row) for row in self.client.embed(texts)]
                self.records[key] = {'texts': texts, 'vectors': vectors}
            except Exception as error:
                self.records[key] = {'texts': texts, 'error_type': type(error).__name__}
        record = self.records[key]
        if record['texts'] != texts:
            self.missing.add(key)
            raise MissingEmbeddingRecording("embedding text mismatch")
        if 'error_type' in record:
            raise ValueError('recorded embedding failure: ' + record['error_type'])
        return record['vectors']

    def snapshot(self):
        return {'model_id': self.model_id, 'records': self.records}

    def verify(self):
        # Production intentionally catches provider exceptions. Replay must still
        # fail on missing recordings instead of counting that as lexical fallback.
        if self.missing:
            raise MissingEmbeddingRecording("missing embedding recordings during replay")
