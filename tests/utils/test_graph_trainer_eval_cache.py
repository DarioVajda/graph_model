"""GraphTrainerV2.get_eval_dataloader must honor the dataset object passed.

HF's Trainer caches the eval dataloader under a constant "eval" key for any
Dataset *object*, so evaluating a test set after the val set (with persistent
workers) otherwise serves the stale val loader — reporting val numbers under a
"test_" prefix. These tests pin the override's cache-invalidation logic without
needing a model: the base Trainer.get_eval_dataloader is spied on to capture the
cache state at the moment our override delegates to it.
"""

from unittest import mock

from src.utils.text_graph_trainer_v2 import GraphTrainerV2


def _bare_trainer():
    # Bypass Trainer.__init__ (needs a model/args); we only exercise the override.
    return GraphTrainerV2.__new__(GraphTrainerV2)


def _spy_on_super():
    """Patch base Trainer.get_eval_dataloader; record the cache keys it sees."""
    seen = {}

    def spy(self, eval_dataset=None):
        seen["keys"] = set(getattr(self, "_eval_dataloaders", {}) or {})
        return "delegated"

    return mock.patch("transformers.Trainer.get_eval_dataloader", spy), seen


def test_object_dataset_drops_stale_eval_cache():
    """A different split (Dataset object) must invalidate the cached 'eval' loader."""
    tr = _bare_trainer()
    tr._eval_dataloaders = {"eval": "VAL_LOADER"}
    patch, seen = _spy_on_super()
    with patch:
        out = tr.get_eval_dataloader(eval_dataset=object())  # e.g. test_dataset
    assert "eval" not in seen["keys"]      # stale entry removed before delegating
    assert out == "delegated"


def test_none_dataset_preserves_cache():
    """In-training evals pass None (fixed val set) -> cache stays, no re-fork."""
    tr = _bare_trainer()
    tr._eval_dataloaders = {"eval": "VAL_LOADER"}
    patch, seen = _spy_on_super()
    with patch:
        tr.get_eval_dataloader(None)
    assert "eval" in seen["keys"]


def test_str_key_dataset_left_to_hf():
    """The named-eval-dataset (str) path is HF's own keyed cache; don't touch it."""
    tr = _bare_trainer()
    tr._eval_dataloaders = {"eval": "VAL_LOADER"}
    patch, seen = _spy_on_super()
    with patch:
        tr.get_eval_dataloader("my_named_set")
    assert "eval" in seen["keys"]


def test_missing_cache_is_safe():
    """No _eval_dataloaders attr yet (first eval) must not raise."""
    tr = _bare_trainer()
    patch, seen = _spy_on_super()
    with patch:
        out = tr.get_eval_dataloader(eval_dataset=object())
    assert out == "delegated"
