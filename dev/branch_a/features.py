"""Frozen pretrained encoders and content-addressed, mmap-compatible caches."""
import os
import resource
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPVisionModel
from .common import check, identity, index_identity, read_json, save_json, sha
from .data import FrameDataset, ProcessorCollator, load_indexes


def device_for(name='auto'):
    if name != 'auto': return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class FrozenEncoder(torch.nn.Module):
    def __init__(self, backbone, checkpoint, revision, cache_dir, device):
        super().__init__()
        self.backbone = backbone
        cls = CLIPVisionModel if backbone == 'clip' else AutoModel
        self.encoder = cls.from_pretrained(checkpoint, revision=revision, cache_dir=cache_dir,
                                           attn_implementation='eager')
        self.encoder.requires_grad_(False); self.encoder.eval(); self.encoder.to(device)
        self.device = device
        self.dim = self.encoder.config.hidden_size
        # Verify final-normalized representations against the actual normalization output.
        norm = self.encoder.vision_model.post_layernorm if backbone=='clip' else self.encoder.layernorm if hasattr(self.encoder,'layernorm') else self.encoder.norm
        self.normalized = None
        self.hook = norm.register_forward_hook(self._capture)
    def _capture(self, module, args, output): self.normalized = output
    def train(self, mode=True):
        super().train(False); self.encoder.eval(); return self
    @torch.inference_mode()
    def forward(self, inputs):
        self.encoder.eval()
        output = self.encoder(**{k:v.to(self.device) for k,v in inputs.items()})
        feature = output.pooler_output if self.backbone=='clip' else output.last_hidden_state[:,0,:]
        reference = self.normalized if self.backbone=='clip' else self.normalized[:,0,:]
        check(feature.ndim==2 and feature.shape[-1]==self.dim, 'Unexpected embedding dimension/output')
        check(torch.allclose(feature,reference), 'Selected features are not the final normalized representation')
        check(not feature.requires_grad and all(not p.requires_grad and p.grad is None for p in self.encoder.parameters()), 'Encoder received gradients')
        return feature


class FeatureCache:
    def __init__(self, path, verify=True):
        self.path = Path(path)
        if (self.path/'latest.json').exists(): self.path = self.path/read_json(self.path/'latest.json')['directory']
        self.meta=read_json(self.path/'metadata.json')
        check(self.meta['cache_identity']==identity(self.meta['identity']), 'Cache identity mismatch')
        if verify: check(sha(self.path/'embeddings.npy')==self.meta['array_sha256'],'Embedding cache corrupted')
        self.values=np.load(self.path/'embeddings.npy',mmap_mode='r')
        self.clips=self.meta['identity']['clips']; self.lookup={c:i for i,c in enumerate(self.clips)}
        check(len(self.clips)==len(self.lookup),'Duplicate cache clips')
        check(self.values.shape==(len(self.clips),8,self.meta['identity']['dimension']), 'Cache shape mismatch')
        check(str(self.values.dtype)==self.meta['identity']['dtype'],'Cache dtype mismatch')
    def rows(self, cids):
        return np.stack([self.values[self.lookup[c]] for c in cids])


def extract(root, split, backbone, config, cache_root, hf_cache, device='auto', batch_size=16, workers=0):
    start=time.monotonic(); torch.set_num_threads(4); videos,frames=load_indexes(root)
    check(split['index_fingerprints']==index_identity(root),'Input indexes differ from splits')
    clips=sorted({r['clip_id'] for rs in split['rounds'].values() for r in rs})
    checkpoint=config['checkpoints'][backbone]
    model_config=AutoConfig.from_pretrained(checkpoint,revision=config.get('revisions',{}).get(backbone,config['revision']),cache_dir=str(hf_cache))
    revision=model_config._commit_hash
    check(revision and len(revision)==40,'Could not resolve immutable model revision')
    # DINOv3 provides only a fast processor; retain the existing CLIP/DINOv2
    # processors so their cached preprocessing identities remain unchanged.
    processor=AutoImageProcessor.from_pretrained(
        checkpoint, revision=revision, cache_dir=str(hf_cache), use_fast=(backbone=='dinov3'))
    dim=model_config.vision_config.hidden_size if backbone=='clip' else model_config.hidden_size
    import transformers
    signature={'checkpoint':checkpoint,'revision':revision,
               'method':'vision_post_layernorm_pooled' if backbone=='clip' else 'final_layernorm_cls',
               'processor':processor.to_dict(),'index_fingerprints':split['index_fingerprints'],
               'clips':clips,'dimension':dim,'dtype':'float32','torch':str(torch.__version__),
               'transformers':transformers.__version__,
               'frame_alignment_fingerprint':identity([[c,[f['sha256'] for f in frames[c]]] for c in clips])}
    key=identity(signature); cache_root=Path(cache_root)/backbone; dest=cache_root/key
    cache_root.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        cached=FeatureCache(dest)
        check(cached.meta['identity']==signature,'Incompatible cache')
        save_json(cache_root/'latest.json',{'directory':key})
        print(f'Reused {backbone} cache: {dest}',flush=True)
        return dest
    target=device_for(device)
    model=FrozenEncoder(backbone,checkpoint,revision,str(hf_cache),target)
    check(model.dim==dim,'Configuration/model dimension mismatch')
    dataset=FrameDataset(root,clips,videos,frames)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,
                      collate_fn=ProcessorCollator(processor))
    with tempfile.TemporaryDirectory(prefix='.partial-',dir=cache_root) as tmp:
        tmp=Path(tmp)
        array=np.lib.format.open_memmap(tmp/'embeddings.npy',mode='w+',dtype='float32',shape=(len(clips),8,dim))
        seen=0
        for batch in loader:
            features=model(batch['inputs']).float().cpu().numpy()
            for item,feature in zip(batch['items'],features):
                check(item['clip_id']==clips[seen//8] and item['frame_position']==seen%8,'Loader/feature alignment mismatch')
                check(np.isfinite(feature).all(),'Nonfinite encoder features')
                array[seen//8,seen%8]=feature;seen+=1
            if seen % (batch_size*10)==0: print(f'{backbone}: {seen}/{len(dataset)} frames encoded',flush=True)
        check(seen==len(dataset),'Incomplete cache')
        array.flush();del array
        elapsed=time.monotonic()-start
        stats={'elapsed_seconds':elapsed,'frames_per_second':seen/elapsed,
               'cache_bytes':(tmp/'embeddings.npy').stat().st_size,
               'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if os.uname().sysname=='Darwin' else 1024),
               'peak_rss_note':'process lifetime high-water mark; includes loading/downloading; shared-process runs are cumulative',
               'device':str(target),'encoder_parameters_frozen':True,'encoder_gradients_none':True}
        save_json(tmp/'metadata.json',{'cache_identity':key,'identity':signature,
                                      'array_sha256':sha(tmp/'embeddings.npy'),'stats':stats})
        os.replace(tmp,dest)
    del model
    if target.type=='mps':torch.mps.empty_cache()
    save_json(cache_root/'latest.json',{'directory':key})
    FeatureCache(dest) # actual reload and checksum/shape verification
    print(f'Created and reloaded {backbone} cache: {dest}',flush=True)
    return dest
