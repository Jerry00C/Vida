"""Explicit model-specific policies; no frozen-feature inference wrappers."""
from dataclasses import dataclass
import hashlib
import torch
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel
from .common import check

@dataclass(frozen=True)
class Layout:
    blocks: str
    norm: str
    attention: str
    query: str
    value: str
    representation: str

LAYOUTS = {
    'clip': Layout('vision_model.encoder.layers','vision_model.post_layernorm','self_attn','q_proj','v_proj','vision_post_layernorm_pooled'),
    'dinov2': Layout('encoder.layer','layernorm','attention.attention','query','value','final_layernorm_cls'),
    'dinov3': Layout('layer','norm','attention','q_proj','v_proj','final_layernorm_cls'),
}


def inspect_structure(encoder, backbone):
    layout=LAYOUTS[backbone]
    blocks=encoder.get_submodule(layout.blocks);norm=encoder.get_submodule(layout.norm)
    check(isinstance(blocks,torch.nn.ModuleList) and len(blocks)==encoder.config.num_hidden_layers,
          'Unexpected transformer block collection/order')
    check(isinstance(norm,torch.nn.LayerNorm),'Expected final LayerNorm')
    targets=[];dim=encoder.config.hidden_size
    for i,block in enumerate(blocks):
        attention=block.get_submodule(layout.attention)
        combined=[n for n,_ in attention.named_modules() if any(x in n.lower() for x in ('qkv','in_proj'))]
        check(not combined,f'Combined QKV requires an explicit new policy: {combined}')
        for projection in (layout.query,layout.value):
            module=attention.get_submodule(projection)
            check(isinstance(module,torch.nn.Linear) and module.weight.shape==(dim,dim),'Unexpected Q/V projection')
            targets.append(f'{layout.blocks}.{i}.{layout.attention}.{projection}')
    check(len(set(targets))==2*len(blocks),'Unexpected LoRA target set')
    return {'blocks':[f'{layout.blocks}.{i}' for i in range(len(blocks))],
            'final_norm':layout.norm,'lora_targets':targets,'combined_qkv':False,
            'dimension':dim,'representation':layout.representation}


class AdaptedVision(torch.nn.Module):
    def __init__(self, encoder, backbone, mode, adaptation):
        super().__init__()
        check(mode in ('frozen','partial','lora'),'Unknown adaptation mode')
        self.backbone=backbone;self.mode=mode;self.adaptation=dict(adaptation)
        self.structure=inspect_structure(encoder,backbone)
        self.encoder=encoder;self.encoder.requires_grad_(False)
        self.unfrozen=[];self.adapter_configuration=None
        if mode=='partial':
            count=adaptation['unfreeze_blocks']
            check(1<=count<=len(self.structure['blocks']),'Invalid number of unfrozen blocks')
            self.unfrozen=self.structure['blocks'][-count:]+[self.structure['final_norm']]
            for name in self.unfrozen:self.encoder.get_submodule(name).requires_grad_(True)
        elif mode=='lora':
            from peft import LoraConfig,get_peft_model
            check(adaptation['bias']=='none' and adaptation['targets']=='query_value_only','Unsupported LoRA targeting policy')
            check(adaptation['rank']>0 and adaptation['alpha']>0 and 0<=adaptation['dropout']<1,'Invalid LoRA settings')
            cfg=LoraConfig(r=adaptation['rank'],lora_alpha=adaptation['alpha'],
                           lora_dropout=adaptation['dropout'],bias='none',
                           target_modules=self.structure['lora_targets'])
            self.encoder=get_peft_model(encoder,cfg)
            actual=set(self.encoder.base_model.targeted_module_names)
            check(actual==set(self.structure['lora_targets']),f'Unexpected PEFT matches: {actual}')
            self.adapter_configuration=cfg.to_dict()
        self.head=torch.nn.Linear(self.structure['dimension'],1)
        self.check_trainable_policy();self.train(False)

    def check_trainable_policy(self):
        names={n for n,p in self.encoder.named_parameters() if p.requires_grad}
        if self.mode=='partial':
            expected={n for n,_ in self.encoder.named_parameters() if any(n.startswith(x+'.') for x in self.unfrozen)}
            check(names==expected and names,'Unexpected partial trainable parameters')
        elif self.mode=='lora':
            check(names and all('.lora_A.' in n or '.lora_B.' in n for n in names),'Original encoder weights unfrozen by LoRA')
            check(len(names)==2*len(self.structure['lora_targets']),'Unexpected LoRA parameter count')
        else:check(not names,'Frozen baseline encoder is trainable')
        check(all(p.requires_grad for p in self.head.parameters()),'Head must train')

    def train(self,mode=True):
        super().train(mode)
        self.encoder.eval()
        if mode and self.mode=='partial':
            for name in self.unfrozen:self.encoder.get_submodule(name).train(True)
        elif mode and self.mode=='lora':
            # Original modules remain in eval; only adapter factors/dropout train.
            for name,module in self.encoder.named_modules():
                if any(part in name.split('.') for part in ('lora_A','lora_B','lora_dropout')):module.train(True)
        self.head.train(mode)
        return self

    def features(self,inputs):
        output=self.encoder(**inputs)
        feature=output.pooler_output if self.backbone=='clip' else output.last_hidden_state[:,0,:]
        check(feature.ndim==2 and feature.shape[-1]==self.structure['dimension'],'Feature dimension mismatch')
        return feature

    def forward(self,inputs):return self.head(self.features(inputs)).squeeze(-1)

    def report(self):
        return {'mode':self.mode,'backbone':self.backbone,'adaptation':self.adaptation,
                'structure':self.structure,'unfrozen_modules':self.unfrozen,
                'lora_targets':self.structure['lora_targets'] if self.mode=='lora' else [],
                'total_parameters':sum(p.numel() for p in self.parameters()),
                'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'trainable_names':[n for n,p in self.named_parameters() if p.requires_grad]}

    def adapted_state(self):
        if self.mode=='lora':
            from peft import get_peft_model_state_dict
            encoder=get_peft_model_state_dict(self.encoder)
        else:encoder={n:p for n,p in self.encoder.named_parameters() if p.requires_grad}
        return {'encoder':{n:p.detach().cpu().clone() for n,p in encoder.items()},
                'head':{n:p.detach().cpu().clone() for n,p in self.head.state_dict().items()}}

    def load_adapted_state(self,state):
        expected=self.adapted_state()
        for section in ('encoder','head'):
            check(set(expected[section])==set(state[section]),'Adapted checkpoint parameter names differ')
            check(all(expected[section][n].shape==v.shape for n,v in state[section].items()),'Adapted checkpoint shape differs')
        if self.mode=='lora':
            from peft import set_peft_model_state_dict
            result=set_peft_model_state_dict(self.encoder,state['encoder'])
            check(not result.unexpected_keys,'Unexpected adapter checkpoint keys')
        else:
            with torch.no_grad():
                for n,p in self.encoder.named_parameters():
                    if p.requires_grad:p.copy_(state['encoder'][n])
        self.head.load_state_dict(state['head'])


def build_model(backbone,mode,base_config,adaptation,hf_cache,device):
    revision=base_config['revisions'][backbone]
    check(len(revision)==40,'An immutable checkpoint revision is required')
    cls=CLIPVisionModel if backbone=='clip' else AutoModel
    encoder=cls.from_pretrained(base_config['checkpoints'][backbone],revision=revision,
                                cache_dir=str(hf_cache),attn_implementation='eager')
    processor=AutoImageProcessor.from_pretrained(base_config['checkpoints'][backbone],revision=revision,
                         cache_dir=str(hf_cache),use_fast=(backbone=='dinov3'))
    model=AdaptedVision(encoder,backbone,mode,adaptation).to(device)
    return model,processor


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class ParameterAudit:
    """Hashes are snapshots of all weights, without duplicating the whole model."""
    def __init__(self,model):
        self.before={n:tensor_hash(p) for n,p in model.named_parameters()}
        self.gradients=set();self.nonzero=set()
    def observe(self,model):
        for n,p in model.named_parameters():
            if not p.requires_grad:check(p.grad is None,f'Frozen gradient: {n}')
            elif p.grad is not None:
                check(torch.isfinite(p.grad).all().item(),f'Nonfinite gradient: {n}')
                self.gradients.add(n)
                if torch.count_nonzero(p.grad).item():self.nonzero.add(n)
    def finish(self,model):
        changed=[n for n,p in model.named_parameters() if tensor_hash(p)!=self.before[n]]
        frozen={n for n,p in model.named_parameters() if not p.requires_grad}
        check(not frozen.intersection(changed),'Frozen encoder weights changed')
        check(any(n.startswith('head.') for n in changed),'Head did not update')
        if model.mode=='partial':
            for module in model.unfrozen:
                prefix='encoder.'+module+'.'
                check(any(n.startswith(prefix) for n in changed),f'Unfrozen module did not update: {module}')
                check(any(n.startswith(prefix) for n in self.nonzero),f'Unfrozen module received no gradient: {module}')
        elif model.mode=='lora':
            for target in model.structure['lora_targets']:
                check(any(target+'.lora_' in n for n in changed),f'Adapter did not update: {target}')
            check(any('.lora_A.' in n for n in changed) and any('.lora_B.' in n for n in changed),'Both LoRA factors must update over multiple steps')
        return {'frozen_unchanged':True,'changed_parameters':changed,'gradient_parameters':sorted(self.gradients),
                'nonzero_gradient_parameters':sorted(self.nonzero),'frozen_parameter_tensors':len(frozen)}
