"""Synthetic focused checks only; the CLI smoke uses real data and pretrained weights."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from dev.common import identity, save_json, sha
from dev.features import FeatureCache
from dev.learning import aggregate_logits, scores_from_logits, metrics, predict, train, load_checkpoint, BalancedFrameSampler
from dev.splits import validate_round, components


def fixture(tmp_path):
    clips=[f'clip{i}' for i in range(16)]
    roles=['train']*8+['validation']*4+['test_seen']*4
    rows=[{'clip_id':c,'round_id':'round_1','role':role,'label':i%2,'source':'real' if i%2==0 else 'gen',
           'component_id':c} for i,(c,role) in enumerate(zip(clips,roles))]
    signature={'clips':clips,'dimension':4,'dtype':'float32','index_fingerprints':{'fixture':'only'},'checkpoint':'synthetic-focused-test'}
    x=np.random.default_rng(1).normal(size=(16,8,4)).astype('float32')
    root=tmp_path/'cache';root.mkdir()
    np.save(root/'embeddings.npy',x)
    save_json(root/'metadata.json',{'identity':signature,'cache_identity':identity(signature),'array_sha256':sha(root/'embeddings.npy')})
    split={'rounds':{'round_1':rows},'smoke':True,'index_fingerprints':signature['index_fingerprints'],'split_identity':'test-split'}
    return root,split,x


def test_aggregation_is_mean_logits_not_mean_scores():
    logits=np.array([[8,0,0,0,0,0,0,0]],dtype=float)
    mean=aggregate_logits(logits)
    assert mean[0]==1
    assert not np.allclose(scores_from_logits(mean),scores_from_logits(logits).mean(axis=1))
    with pytest.raises(ValueError):aggregate_logits(np.ones((2,7)))


def test_one_class_metrics():
    m=metrics([0,0],[.1,.2])
    assert m['auroc'] is None and m['recall'] is None and m['precision'] is None
    assert m['confusion_matrix']==[[2,0],[0,0]]
    assert metrics([0,1],[.49,.5])['f1']==1


def test_leakage_rejected():
    videos={'a':{'family':'G','label':'1'},'b':{'family':'G','label':'1'}}
    rows=[{'clip_id':c,'role':role,'label':1,'source':'G','component_id':'shared'} for c,role in [('a','train'),('b','validation')]]
    with pytest.raises(ValueError,match='leaks'):validate_round(rows,['H'],videos)
    rows[0]['source']='H';videos['a']['family']='H'
    with pytest.raises(ValueError,match='isolation'):validate_round(rows,['H'],videos)


def test_transitive_duplicates():
    videos={'a':{'group_id':'g1','sha256':'sha1'},'b':{'group_id':'g1','sha256':'sha2'},'c':{'group_id':'g2','sha256':'sha2'}}
    frames={c:[{'sha256':c+str(i)} for i in range(8)] for c in videos}
    groups=components(videos,frames)
    assert len(set(groups.values()))==1


def test_feature_prediction_alignment(tmp_path):
    path,split,x=fixture(tmp_path);cache=FeatureCache(path)
    rows=list(reversed(split['rounds']['round_1']))
    head=torch.nn.Linear(4,1)
    with torch.no_grad():head.weight.fill_(1);head.bias.zero_()
    p=predict(head,cache,rows,3)
    for row,result in zip(rows,p):
        assert result['clip_id']==row['clip_id']
        expected=x[cache.lookup[row['clip_id']]].sum(axis=1)
        np.testing.assert_allclose(result['frame_logits'],expected,rtol=1e-6,atol=1e-6)


def test_cache_corruption(tmp_path):
    path,_,_=fixture(tmp_path)
    with (path/'embeddings.npy').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='corrupted'):FeatureCache(path)


def test_balanced_sampling_reproducible(tmp_path):
    _,split,_=fixture(tmp_path);rows=split['rounds']['round_1']
    a=list(BalancedFrameSampler(rows,42,1));b=list(BalancedFrameSampler(rows,42,1))
    assert a==b and all(0<=frame<8 for _,frame in a)
    assert a!=list(BalancedFrameSampler(rows,42,2))


def test_checkpoint_resume_exact_and_identity(tmp_path):
    path,split,_=fixture(tmp_path)
    cfg={'seed':42,'epochs':2,'lr':1e-4,'weight_decay':1e-4,'batch_size':4,
         'patience':5,'min_delta':.0001,'scheduler_factor':.5,'scheduler_patience':2,
         'balanced_classes':True,'equal_generators':True}
    train(path,split,'round_1',cfg,tmp_path/'control')
    train(path,split,'round_1',cfg,tmp_path/'resume',stop_after=1)
    train(path,split,'round_1',cfg,tmp_path/'resume',resume=True)
    cache=FeatureCache(path)
    a=load_checkpoint(tmp_path/'control/latest.pt',cache,'test-split','round_1')
    b=load_checkpoint(tmp_path/'resume/latest.pt',cache,'test-split','round_1')
    assert a['history']==b['history']
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert a['optimizer']['param_groups'][1]['weight_decay']==0
    with pytest.raises(ValueError,match='split'):load_checkpoint(tmp_path/'resume/best.pt',cache,'changed','round_1')


def test_frozen_encoder_no_gradients(monkeypatch):
    from transformers import Dinov2Config,Dinov2Model
    import dev.features as module
    model=Dinov2Model(Dinov2Config(hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,image_size=28,patch_size=14))
    monkeypatch.setattr(module.AutoModel,'from_pretrained',lambda *a,**k:model)
    frozen=module.FrozenEncoder('dinov2','synthetic-fixture','fixture',None,torch.device('cpu'))
    frozen.train()
    embedding=frozen({'pixel_values':torch.rand(2,3,28,28)})
    assert embedding.shape==(2,16) and not embedding.requires_grad and not frozen.encoder.training
    head=torch.nn.Linear(16,1);head(embedding.clone()).sum().backward()
    assert all(p.grad is None for p in frozen.encoder.parameters())


def test_heldout_connected_test_member_rejected():
    videos={'a':{'family':'Held','label':'1'},'b':{'family':'Seen','label':'1'}}
    rows=[{'clip_id':'a','role':'test_unseen','label':1,'source':'Held','component_id':'shared'},
          {'clip_id':'b','role':'test_seen','label':1,'source':'Seen','component_id':'shared'}]
    with pytest.raises(ValueError,match='heldout-connected'):validate_round(rows,['Held'],videos)


def test_round_initialization_and_resume(tmp_path):
    path,split,_=fixture(tmp_path)
    split['rounds']['round_2']=copy.deepcopy(split['rounds']['round_1'])
    cfg={'seed':42,'epochs':1,'lr':0.,'weight_decay':0.,'batch_size':4,
         'patience':5,'min_delta':.0001,'scheduler_factor':.5,'scheduler_patience':2,
         'balanced_classes':True,'equal_generators':True}
    first=train(path,split,'round_1',cfg,tmp_path/'first')
    first['model']['weight'].fill_(3.)
    source=tmp_path/'initial.pt';torch.save(first,source)
    second=train(path,split,'round_2',cfg,tmp_path/'second',init_checkpoint=source)
    assert torch.equal(second['model']['weight'],first['model']['weight'])
    assert second['epoch']==1 and second['round_id']=='round_2'
    resumed=train(path,split,'round_2',cfg,tmp_path/'second',resume=True,init_checkpoint=source)
    assert torch.equal(resumed['model']['weight'],second['model']['weight'])
    with pytest.raises(ValueError,match='configuration'):
        train(path,split,'round_2',cfg,tmp_path/'second',resume=True)
