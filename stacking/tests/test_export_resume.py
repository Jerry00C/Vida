import json
import pytest
from stacking.export_b import read_cache


def test_interrupted_tail_recovers_completed_records(tmp_path):
    path=tmp_path/"progress.jsonl"
    complete=(json.dumps({"clip_id":"one","raw_margin":0.5})+"\n").encode()
    path.write_bytes(complete+b'{"clip_id":"interrupted')
    assert read_cache(path)==[{"clip_id":"one","raw_margin":0.5}]
    assert path.read_bytes()==complete


def test_corrupt_middle_record_is_not_silently_discarded(tmp_path):
    path=tmp_path/"progress.jsonl"
    path.write_bytes(b'{"clip_id":"first"}\ncorrupt\n{"clip_id":"last"}\n')
    with pytest.raises(ValueError,match="Corrupt non-tail"):
        read_cache(path)
