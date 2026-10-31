"""Synthetic small models: regression tests, distinct from real smoke runs."""
import copy
import pytest
import torch
from transformers import CLIPVisionConfig,CLIPVisionModel,Dinov2Config,Dinov2Model,DINOv3ViTConfig,DINOv3ViTModel
from dev.adapters import AdaptedVision,ParameterAudit
from dev.common import read_json,identity
from dev.image_learning import optimizer_for,window_samples,precision_policy,load_adapted_checkpoint

CFG=read_json('dev/adaptation_config.json')
torch.set_num_threads(2)

def encoder(name):
    kwargs=dict(hidden_size=32,num_hidden_layers=3,num_attention_heads=4,intermediate_size=64,image_size=32,patch_size=16)
    classes={'clip':(CLIPVisionConfig,CLIPVisionModel),'dinov2':(Dinov2Config,Dinov2Model),'dinov3':(DINOv3ViTConfig,DINOv3ViTModel)}
    config,model=classes[name]
    return model(config(**kwargs))

@pytest.mark.parametrize('name',['clip','dinov2','dinov3'])
@pytest.mark.parametrize('mode',['partial','frozen'])
def test_policy_gradients_reload(name,mode):
    torch.manual_seed(7)
    base=encoder(name);original=copy.deepcopy(base)
    model=AdaptedVision(base,name,mode,CFG['adaptation'])
    inputs={'pixel_values':torch.randn(2,3,32,32)}
    original.eval();model.eval()
    with torch.no_grad():
        out=original(**inputs)
        expected=out.pooler_output if name=='clip' else out.last_hidden_state[:,0]
        torch.testing.assert_close(model.features(inputs),expected,rtol=0,atol=0)
    model.train();model.check_trainable_policy()
    if mode=='partial':
        assert not model.encoder.get_submodule(model.structure['blocks'][0]).training
        assert all(model.encoder.get_submodule(n).training for n in model.unfrozen)
    audit=ParameterAudit(model);opt=optimizer_for(model,CFG['training'])
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.binary_cross_entropy_with_logits(model(inputs),torch.tensor([0.,1.])).backward()
        audit.observe(model);opt.step()
    assert audit.finish(model)['frozen_unchanged']
    for group in opt.param_groups:
        if any(n.endswith('.bias') for n in group['parameter_names']):assert group['weight_decay']==0
    model.eval();before=model(inputs).detach()
    rebuilt=AdaptedVision(original,name,mode,CFG['adaptation']);rebuilt.load_adapted_state(model.adapted_state());rebuilt.eval()
    torch.testing.assert_close(before,rebuilt(inputs),rtol=0,atol=0)
    assert all(not m.training for m in rebuilt.modules())

@pytest.mark.parametrize('samples,batch,accumulation',[(33,4,2),(11,3,3),(3,8,2)])
def test_incomplete_accumulation(samples,batch,accumulation):
    torch.manual_seed(1);x=torch.randn(samples,3);y=torch.randn(samples,1)
    a=torch.nn.Linear(3,1);b=copy.deepcopy(a)
    oa=torch.optim.SGD(a.parameters(),lr=.1);ob=torch.optim.SGD(b.parameters(),lr=.1)
    batches=(samples+batch-1)//batch;oa.zero_grad()
    for i,start in enumerate(range(0,samples,batch)):
        loss=(a(x[start:start+batch])-y[start:start+batch]).square().sum()
        (loss/window_samples(i,batches,samples,batch,accumulation)).backward()
        if (i+1)%accumulation==0 or i+1==batches:oa.step();oa.zero_grad()
    for start in range(0,samples,batch*accumulation):
        ob.zero_grad();(b(x[start:start+batch*accumulation])-y[start:start+batch*accumulation]).square().mean().backward();ob.step()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)


def test_identity_rejection_and_precision(tmp_path):
    path=tmp_path/'checkpoint.pt';spec={'mode':'partial','seed':42}
    torch.save({'spec':spec,'experiment_identity':identity(spec)},path)
    assert load_adapted_checkpoint(path)['spec']==spec
    with pytest.raises(ValueError,match='Incompatible'):load_adapted_checkpoint(path,'wrong')
    assert precision_policy(torch.device('cpu'),'auto')=='fp32'
    with pytest.raises(ValueError,match='CUDA'):precision_policy(torch.device('cpu'),'fp16')


def test_image_sampling_worker_reproducibility(tmp_path):
    from PIL import Image
    from transformers import CLIPImageProcessor
    from dev.common import sha
    from dev.image_learning import make_loader
    rows=[];videos={};frames={}
    for i in range(8):
        cid=f'clip{i}';label=i%2;source='fake' if label else 'real'
        rows.append(dict(clip_id=cid,label=label,source=source));videos[cid]={'label':str(label),'family':source};frames[cid]=[]
        for j in range(8):
            path=tmp_path/f'{cid}_{j}.png';Image.new('RGB',(32,32),(i*20,j*20,0)).save(path)
            frames[cid].append({'relative_frame_path':path.name,'sha256':sha(path)})
    processor=CLIPImageProcessor(size={'shortest_edge':32},crop_size={'height':32,'width':32})
    def collect(workers,epoch):
        result=[]
        for batch in make_loader(tmp_path,rows,videos,frames,processor,3,workers,42,epoch,CFG['training']):
            result.extend((r['clip_id'],r['frame_position'],r['label']) for r in batch['items'])
        return result
    assert collect(0,1)==collect(2,1)
    assert collect(0,1)!=collect(0,2)
