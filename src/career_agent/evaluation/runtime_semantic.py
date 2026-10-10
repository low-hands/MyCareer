"""Per-sample capability score tape, including explicit lexical fallback.

Scores are inputs to runtime selection, not predictions produced during replay.
Missing queries or catalogue mismatches fail replay rather than call a provider.
"""
from career_agent.evaluation.capability_semantic_fixture import catalogue_digest


class SemanticTape:
    def __init__(self, scorer=None, *, recording=None):
        self.scorer = scorer
        self.replaying = recording is not None
        self.queries = {} if recording is None else dict(recording["queries"])
        self.missing = []
        if recording is not None and recording["catalogue_digest"] != catalogue_digest():
            raise ValueError("recorded semantic catalogue changed")

    def scores(self, query):
        if query not in self.queries:
            if self.replaying:
                self.missing.append(query)
                raise ValueError("semantic query missing from recording")
            try:
                scores = None if self.scorer is None else dict(self.scorer.scores(query))
            except Exception:
                scores = None
            self.queries[query] = scores
        scores = self.queries[query]
        if scores is None:
            # Both production consumers already handle embedding failures by
            # using lexical ranking. Reproduce that exact recorded fallback.
            raise ValueError("recorded lexical fallback")
        return scores

    def snapshot(self):
        return {"catalogue_digest": catalogue_digest(), "queries": self.queries}

    def verify(self):
        if self.missing:
            raise ValueError("semantic queries missing from recording")
