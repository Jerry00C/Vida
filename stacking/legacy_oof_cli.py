import argparse
from .core import BRANCHES, aligned, make_plan, read, template, write


def main():
    p = argparse.ArgumentParser(description='Three-branch video score stacking (run from PR2)')
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('folds')
    c.add_argument('--split', required=True); c.add_argument('--round', default='round_1')
    c.add_argument('--folds', type=int, default=5); c.add_argument('--seed', type=int, default=42)
    c.add_argument('--output', required=True)
    c = sub.add_parser('template')
    c.add_argument('--plan', required=True); c.add_argument('--branch', choices=BRANCHES, required=True)
    c.add_argument('--output', required=True)
    c = sub.add_parser('branch-a')
    c.add_argument('--plan', required=True); c.add_argument('--split', required=True)
    c.add_argument('--cache', required=True); c.add_argument('--output', required=True)
    c.add_argument('--config', default='branches_dev/branch_a/dev/config.json'); c.add_argument('--resume', action='store_true')
    c = sub.add_parser('export-a')
    c.add_argument('--cache', required=True); c.add_argument('--checkpoint', required=True)
    c.add_argument('--hf-cache', default='branches_dev/branch_a/runs/branch_a/huggingface'); c.add_argument('--output', required=True)
    for name in ('validate', 'fit', 'evaluate'):
        c = sub.add_parser(name)
        c.add_argument('--plan', required=True)
        for b in BRANCHES: c.add_argument('--'+b.lower(), required=True)
        if name != 'validate': c.add_argument('--output', required=True)
        if name == 'fit': c.add_argument('--C', nargs='+', type=float, default=[.01, .1, 1., 10.])
        if name == 'evaluate': c.add_argument('--model', required=True)
    a = p.parse_args()
    if a.command == 'folds':
        write(a.output, make_plan(read(a.split), a.round, a.folds, a.seed))
    elif a.command == 'template':
        write(a.output, template(read(a.plan), a.branch))
    elif a.command == 'branch-a':
        from .branch_a import train_scores
        train_scores(read(a.plan), read(a.split), a.cache, read(a.config)['training'], a.output, a.resume)
    elif a.command == 'export-a':
        from .branch_a import export_frozen
        export_frozen(a.cache, a.checkpoint, a.hf_cache, a.output)
    else:
        plan = read(a.plan)
        bundles = {b: read(getattr(a, b.lower())) for b in BRANCHES}
        if a.command == 'validate':
            data = aligned(plan, bundles)
            print({role: len(rows) for role, (rows, _, _) in data.items()})
        elif a.command == 'fit':
            from .fusion import fit
            fit(plan, bundles, a.output, a.C)
        else:
            from .fusion import evaluate
            evaluate(plan, bundles, read(a.model), a.output)


if __name__ == '__main__':
    main()
