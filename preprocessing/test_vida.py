import json
import zipfile
from pathlib import Path

import av
from PIL import Image
import pytest

import vida


def fixture_row(tmp_path, count=8):
    delivery = tmp_path / 'delivery'
    delivery.mkdir()
    source = tmp_path / 'source.mkv'
    with av.open(str(source), 'w') as container:
        stream = container.add_stream('ffv1', rate=8)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = 'bgr0'
        for i in range(count):
            frame = av.VideoFrame.from_image(Image.new('RGB', (64, 64), (i*20, 30, 100)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    archive = delivery / 'videos.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.write(source, 'videos/Test/clip.mkv')
    row = {'clip_id': 'clip', 'filesystem_key': 'clip', 'label': '0', 'family': 'Test',
           'class': 'real', 'group_id': 'parent', 'split': 'train', 'fold': '0',
           'validation_role': '', 'frame_count': str(count), 'file_bytes': str(source.stat().st_size),
           'sha256': vida.digest(source), 'archive_path': 'videos.zip',
           'archive_member_path': 'videos/Test/clip.mkv',
           'organized_video_path': 'real/Test/videos/clip.mkv',
           'frame_directory_path': 'real/Test/frames/clip'}
    return delivery, row


def test_sampling():
    assert vida.sample_indices(8) == list(range(8))
    assert vida.sample_indices(16) == [0, 2, 4, 6, 8, 10, 12, 15]
    for n in range(8, 2000):
        indices = vida.sample_indices(n)
        assert len(set(indices)) == 8 and indices == sorted(indices)
        assert indices[0] == 0 and indices[-1] == n-1
    with pytest.raises(ValueError):
        vida.sample_indices(7)


@pytest.mark.parametrize('path', ['../evil', '/evil', 'x/../../evil', 'C:/evil', 'x\\evil'])
def test_path_rejection(path):
    with pytest.raises(ValueError):
        vida.safe_relative(path)


def test_safe_key():
    assert vida.safe_key('vript_--abc-Scene-045') == 'vript_--abc-Scene-045'
    assert '/' not in vida.safe_key('../bad')
    assert vida.safe_key('../bad') != vida.safe_key('../BAD')


def test_resume_and_repair(tmp_path):
    delivery, row = fixture_row(tmp_path)
    output = tmp_path / 'processed'
    config = vida.settings()
    record, skipped = vida.process(row, delivery, output, config, True)
    assert not skipped and record['video']['frame_extraction_status'] == 'success', record['errors']
    assert vida.valid_record(output, row, config, record)
    again, skipped = vida.process(row, delivery, output, config, True)
    assert skipped and again == record
    # A readable but altered PNG must not be skipped.
    image = output / record['frames'][0]['relative_frame_path']
    Image.new('RGB', (64,64), (255,0,0)).save(image)
    repaired, skipped = vida.process(row, delivery, output, config, True)
    assert not skipped and vida.valid_record(output, row, config, repaired)
    # Simulate interruption after publication but before committing its record.
    vida.record_path(output, row).unlink()
    recovered, skipped = vida.process(row, delivery, output, config, True)
    assert not skipped and vida.valid_record(output, row, config, recovered)
    # Missing PNG is incomplete even when the record exists.
    (output / recovered['frames'][1]['relative_frame_path']).unlink()
    recovered, skipped = vida.process(row, delivery, output, config, True)
    assert not skipped and vida.valid_record(output, row, config, recovered)
    vida.publish(output, [recovered])
    vida.publish(output, [recovered])
    assert len(vida.read_csv(output / 'indexes/videos.csv')) == 1
    assert len(vida.read_csv(output / 'indexes/frames.csv')) == 8
    bad = json.loads(json.dumps(recovered))
    bad['frames'][0]['clip_id'] = 'wrong'
    with pytest.raises(ValueError):
        vida.valid_record(output, row, config, bad)


def test_short_clip_failure(tmp_path):
    delivery, row = fixture_row(tmp_path, 7)
    record, _ = vida.process(row, delivery, tmp_path/'output', vida.settings(), True)
    assert record['video']['video_extraction_status'] == 'success'
    assert record['video']['frame_extraction_status'] == 'failed'
    assert not record['frames'] and 'need at least eight' in record['errors'][0]['error']


def test_existing_video_not_overwritten(tmp_path):
    delivery, row = fixture_row(tmp_path)
    output = tmp_path/'output'
    dest = output/row['organized_video_path']
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'wrong bytes')
    record, _ = vida.process(row, delivery, output, vida.settings(), True)
    assert record['video']['video_extraction_status'] == 'failed'
    assert dest.read_bytes() == b'wrong bytes'


def test_resumable_migration(tmp_path, monkeypatch):
    dev = tmp_path/'dev'
    dev.mkdir()
    (dev/'one.txt').write_bytes(b'original bytes')
    (dev/'packages').mkdir()
    (dev/'packages'/'a.zip').write_bytes(b'archive bytes')
    (dev/'sources').mkdir()
    (dev/'sources'/'synced.txt').write_bytes(b'untouched')
    monkeypatch.setattr(vida, 'DELIVERY_ITEMS', ['one.txt', 'packages'])
    inventory = {'items': ['one.txt', 'packages'], 'files': [
        {'path': str(p.relative_to(dev)), 'bytes': p.stat().st_size, 'sha256': vida.digest(p)}
        for p in [dev/'one.txt', dev/'packages'/'a.zip']]}
    vida.atomic_json(dev/'migration_inventory.json', inventory)
    target = dev/'preprocess-from Zhou'
    target.mkdir()
    (dev/'one.txt').rename(target/'one.txt')
    vida.migrate(dev)
    vida.migrate(dev)
    assert (target/'one.txt').read_bytes() == b'original bytes'
    assert (target/'packages'/'a.zip').read_bytes() == b'archive bytes'
    assert (dev/'sources'/'synced.txt').read_bytes() == b'untouched'
