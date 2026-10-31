"""Protocol isolation and accelerated bootstrap equivalence checks."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from branch_c.evaluation import bootstrap_intervals

spec=importlib.util.spec_from_file_location("generator_rotation",Path(__file__).parents[1]/"scripts/generator_rotation.py")
rotation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(rotation)


def test_group_partitions_are_exact_stable_and_fold_balanced():
    strata={"parent_pairs":[f"p{i}" for i in range(100)],"prompts":[f"q{i}" for i in range(50)]}
    a=rotation.partition_groups(strata)
    b=rotation.partition_groups({k:list(reversed(v)) for k,v in reversed(list(strata.items()))})
    assert a==b
    for ids in strata.values():
        roles=[a[g][0] for g in ids]
        assert [roles.count(r) for r in ['train','model_selection','calibration','test']]==[int(.6*len(ids)),int(.1*len(ids)),int(.1*len(ids)),int(.2*len(ids))]
        folds=[a[g][1] for g in ids if a[g][0]=='train']
        assert len(set(folds.count(str(f)) for f in range(5)))==1


def test_quarantine_is_per_round_not_permanently_heldout():
    rows=[]
    for g in rotation.GENERATORS:
        for split in ['train','val','test']:
            rows.append({'clip_id':g+split,'generator':g,'split':split,'label':'1','protocol':'base','rotation_eligible':'true'})
    a,excluded=rotation.round_records(rows,('SVD','MuseV'),'r1')
    b,_=rotation.round_records(rows,('Pika','OpenSora'),'r2')
    assert not any(r['generator'] in {'SVD','MuseV'} and r['split']!='test' for r in a)
    assert any(r['generator']=='SVD' and r['split']=='train' for r in b)
    assert sum(r['evaluation_subset']=='unseen_generators' for r in a if r['split']=='test')==2
    assert len(excluded)==4


def test_partial_parent_quota_is_rejected():
    class R:
        source='Vript'
    groups={'a':[R(),R()],'b':[R(),R()]}
    with pytest.raises(ValueError,match='whole-group quota'):
        rotation.take_whole(groups,set(groups),3,'Vript','test')


@pytest.mark.parametrize('tied',[False,True])
@pytest.mark.parametrize('threshold',[-.5,.3])
def test_weighted_group_bootstrap_matches_reference_resampling(tied,threshold):
    rng=np.random.default_rng(19)
    y=np.tile([0,1],30)
    s=rng.normal(size=60)
    if tied:s=np.round(s,0)
    groups=np.repeat(np.arange(30),2)
    reference=bootstrap_intervals(y,s,groups,threshold,repeats=137,seed=42)
    actual=rotation.fast_group_bootstrap(y,s,groups,threshold,repeats=137,seed=42)
    assert actual['valid_repeats']==reference['valid_repeats']==137
    assert actual['n_groups']==reference['n_groups']
    np.testing.assert_allclose(actual['auroc'],reference['auroc'],atol=1e-14,rtol=0)
    np.testing.assert_allclose(actual['macro_f1'],reference['macro_f1'],atol=1e-14,rtol=0)


def test_bootstrap_rejects_nonfinite_and_single_class():
    with pytest.raises(ValueError):rotation.fast_group_bootstrap([0,1],[0,np.nan],[0,1],.5)
    with pytest.raises(ValueError):rotation.fast_group_bootstrap([0,0],[0,1],[0,1],.5)
