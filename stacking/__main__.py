import argparse
import json
from pathlib import Path
from . import pipeline as p

def main():
    parser=argparse.ArgumentParser(description="Train only the VIDA scalar stacking layer on independent holdout scores")
    parser.add_argument("--config",default=str(Path(__file__).with_name("config.local.json")))
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("doctor","prepare-requests","train"):sub.add_parser(name)
    e=sub.add_parser("evaluate");e.add_argument("--bootstrap",type=int,default=200)
    i=sub.add_parser("import-scores");i.add_argument("--branch",choices=p.BRANCHES,required=True);i.add_argument("--input",required=True)
    identity=i.add_mutually_exclusive_group(required=True);identity.add_argument("--model-id");identity.add_argument("--checkpoint")
    i.add_argument("--score-column");i.add_argument("--direction",choices=("higher_is_fake","higher_is_real"),default="higher_is_fake")
    i.add_argument("--confirm-base-only",action="store_true",help="Confirm frozen model did not use new holdouts in fitting/selection")
    d=sub.add_parser("demo");d.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="demo":
        from .demo import run
        result=run(Path(args.output))
    else:
        cfg=p.config(args.config)
        if args.command=="doctor":result=p.doctor(cfg)
        elif args.command=="prepare-requests":result=p.prepare_requests(cfg)
        elif args.command=="train":result=p.train(cfg)
        elif args.command=="evaluate":result=p.evaluate(cfg,args.bootstrap)
        else:
            p.require(args.confirm_base_only,"Confirm actual branch training provenance using --confirm-base-only")
            result=p.import_scores(cfg,args.branch,args.input,args.model_id or p.digest(args.checkpoint),args.score_column,args.direction)
    if args.command=='train' and result.get('neural_candidate'):
        result=dict(result);candidate=dict(result['neural_candidate'])
        candidate['folds']=[{k:v for k,v in f.items() if k!='history'} for f in candidate['folds']]
        candidate.pop('final_training_history',None);result['neural_candidate']=candidate
        result['full_epoch_history']='Saved in output_dir/training_report.json'
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=="__main__":main()
