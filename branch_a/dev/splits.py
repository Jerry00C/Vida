"""Connected-group partitions; holdout isolation overrides target ratios."""
from collections import Counter, defaultdict
from pathlib import Path
from .common import check, identity, read_json, save_csv, save_json
from .data import audited, load_indexes


class UnionFind:
    def __init__(self, ids): self.parent = {x:x for x in ids}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a, b):
        a,b = self.find(a),self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)


def components(videos, frames):
    uf = UnionFind(videos)
    seen = {}
    for cid, v in videos.items():
        keys = [('group', v['group_id']), ('video_sha', v['sha256']),
                ('eight_frame_sha', identity([f['sha256'] for f in frames[cid]]))]
        check(v['group_id'], 'Missing leakage group')
        for key in keys:
            if key in seen: uf.union(cid, seen[key])
            else: seen[key] = cid
    return {cid: uf.find(cid) for cid in videos}


def assign_base(videos, comp, seed):
    groups = defaultdict(list)
    for cid in sorted(videos): groups[comp[cid]].append(cid)
    # Components with the same source-count vector form strata. Entire components
    # move together. Real assignments are made once, then reused in every round.
    strata = defaultdict(list)
    for group, ids in groups.items():
        signature = tuple(sorted(Counter(videos[c]['family'] for c in ids).items()))
        strata[signature].append(group)
    assignments = {}
    for signature, gs in sorted(strata.items()):
        gs.sort(key=lambda g: identity([seed, g]))
        n = len(gs); train = round(.7*n); val = round(.15*n)
        for i,g in enumerate(gs): assignments[g] = 'train' if i < train else 'validation' if i < train+val else 'test'
    return assignments


def validate_round(rows, heldout, videos):
    check(len({(r['clip_id'], r['role']) for r in rows}) == len(rows), 'Duplicate split row')
    roles = defaultdict(set); clips = defaultdict(set)
    for r in rows:
        roles[r['component_id']].add(r['role']); clips[r['clip_id']].add(r['role'])
        v = videos[r['clip_id']]
        check(v['family'] == r['source'] and int(v['label']) == r['label'], 'Split metadata mismatch')
        if r['label'] == 1:
            check((r['source'] in heldout) == (r['role'] == 'test_unseen'), 'Generator isolation failure')
    for values in roles.values():
        check(len(values) == 1 or values == {'test_seen','test_unseen'}, 'Connected group leaks across train/validation/test')
    for cid, values in clips.items():
        check(len(values) == 1 or (videos[cid]['label']=='0' and values == {'test_seen','test_unseen'}), 'Clip overlap')
    held_components = {r['component_id'] for r in rows if r['source'] in heldout}
    check(all(r['source'] in heldout for r in rows if r['component_id'] in held_components),
          'Non-heldout clip retained in heldout-connected component')
    real_seen = {r['clip_id'] for r in rows if r['role']=='test_seen' and r['label']==0}
    real_unseen = {r['clip_id'] for r in rows if r['role']=='test_unseen' and r['label']==0}
    check(real_seen == real_unseen, 'Test negative examples must be shared')


def build(root, audit_path, config, output):
    aud = audited(root, audit_path)
    all_v, frames = load_indexes(root)
    videos = {c:all_v[c] for c in aud['eligible']}
    generators = sorted({v['family'] for v in videos.values() if v['label']=='1'})
    pairs = config['pairs']
    check(len(pairs)==4 and all(len(p)==2 for p in pairs) and sorted(x for p in pairs for x in p)==generators,
          'Pairs must hold each of eight generators out exactly once')
    comp = components(videos, frames); base = assign_base(videos, comp, config['seed'])
    rounds, exclusions, counts = {}, [], {}
    for i,pair in enumerate(pairs,1):
        rid = f'round_{i}'
        blocked = {comp[c] for c,v in videos.items() if v['family'] in pair}
        rows = []
        for cid,v in sorted(videos.items()):
            label = int(v['label']); group = comp[cid]
            if v['family'] in pair: roles = ['test_unseen']
            elif group in blocked:
                exclusions.append({'round_id':rid,'clip_id':cid,'source':v['family'],
                                   'component_id':group,'reason':'connected_to_heldout_generator'})
                continue
            elif base[group]=='test': roles = ['test_seen','test_unseen'] if label==0 else ['test_seen']
            else: roles = [base[group]]
            for role in roles:
                rows.append({'round_id':rid,'clip_id':cid,'role':role,'label':label,'source':v['family'],
                             'component_id':group,'original_split':v['split'],'original_fold':v['fold'],
                             'group_id':v['group_id'],'original_source_id':v['original_source_id'],
                             'validation_role':v['validation_role']})
        validate_round(rows,pair,videos)
        held_ids = {c for c,v in videos.items() if v['family'] in pair}
        check(held_ids == {r['clip_id'] for r in rows if r['role']=='test_unseen' and r['label']==1}, 'Heldout coverage mismatch')
        rounds[rid]=rows
        counts[rid]=dict(Counter(f'{r["role"]}/{r["label"]}/{r["source"]}' for r in rows))
    result = {'version':config['version'],'seed':config['seed'],'pairs':pairs,
              'pairing_status':config['pairing_status'],'index_fingerprints':aud['index_fingerprints'],
              'audit_identity':aud['audit_identity'],'rounds':rounds,'exclusions':exclusions,'counts':counts,
              'group_policy':'transitive union of group_id, exact video SHA256, and ordered eight-frame SHA256 signature; exclude all non-heldout members of holdout-connected components',
              'shared_real_test_examples':True,'base_assignments':base,'smoke':False}
    result['split_identity']=identity(result)
    output = Path(output)
    if output.exists(): check(read_json(output)==result,'Split version already exists with different content; use a new output path')
    save_json(output,result)
    all_rows=[r for rs in rounds.values() for r in rs]
    save_csv(output.with_suffix('.csv'),all_rows,list(all_rows[0]))
    save_csv(output.with_suffix('.exclusions.csv'),exclusions,['round_id','clip_id','source','component_id','reason'])
    print('All four rounds pass leakage checks. Excluded rows:',len(exclusions),flush=True)
    return result


def load_split(path):
    s=read_json(path); expected=dict(s); expected.pop('split_identity')
    check(identity(expected)==s['split_identity'],'Split content changed')
    return s


def smoke_subset(split, output, videos):
    rid='round_1'; rows=split['rounds'][rid]; picked=[]
    # Source-stratified component selection, whole related groups retained.
    for role, target in [('train',16),('validation',8),('test_seen',8)]:
        for label in (0,1):
            candidate=[r for r in rows if r['role']==role and r['label']==label]
            groups=defaultdict(list)
            for r in candidate: groups[r['component_id']].append(r)
            buckets=defaultdict(list)
            for g,rs in groups.items(): buckets[tuple(sorted({r['source'] for r in rs}))].append(g)
            for gs in buckets.values():gs.sort(key=lambda g:identity([42,role,label,g]))
            count=0
            while count < target and buckets:
                for key in sorted(list(buckets)):
                    g=buckets[key].pop(0); picked.extend(groups[g]); count+=len(groups[g])
                    if not buckets[key]:del buckets[key]
                    if count>=target:break
    real_test=[dict(r,role='test_unseen') for r in picked if r['role']=='test_seen' and r['label']==0]
    picked.extend(real_test)
    for generator in split['pairs'][0]:
        candidates=[r for r in rows if r['role']=='test_unseen' and r['source']==generator]
        groups=defaultdict(list)
        for r in candidates:groups[r['component_id']].append(r)
        n=0
        for g in sorted(groups,key=lambda g:identity([42,g])):
            picked.extend(groups[g]); n+=len(groups[g])
            if n>=4:break
    validate_round(picked,split['pairs'][0],videos)
    for role in ('train','validation','test_seen','test_unseen'):
        check({r['label'] for r in picked if r['role']==role}=={0,1},'Smoke role lacks both classes')
    result={'version':'smoke-verification-v1','parent_split_identity':split['split_identity'],
            'pairs':split['pairs'],'index_fingerprints':split['index_fingerprints'],
            'rounds':{rid:picked},'smoke':True,'shared_real_test_examples':True}
    result['split_identity']=identity(result)
    save_json(output,result);save_csv(Path(output).with_suffix('.csv'),picked,list(picked[0]))
    print('Smoke counts:',dict(Counter(r['role'] for r in picked)),flush=True)
    return result
