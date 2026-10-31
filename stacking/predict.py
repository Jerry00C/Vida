"""Apply a frozen stacker to already computed A/B/C scores for new videos."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from . import pipeline as p

def main():
    parser=argparse.ArgumentParser();source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config',help='Original V1 configuration')
    source.add_argument('--model-dir',help='Portable V1/V2 folder containing stacker.joblib and MODEL_LOCK.json')
    parser.add_argument('--input',required=True,help='CSV: clip_id,A,B,C, all higher-is-generated raw scores')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    cfg={'output_dir':str(Path(args.model_dir).resolve())} if args.model_dir else p.config(args.config)
    bundle=p.load_bundle(cfg);frame=p.table(args.input)
    p.require({'clip_id',*p.BRANCHES}.issubset(frame),'Expected clip_id,A,B,C columns')
    p.require(len(frame)>0 and frame.clip_id.is_unique,'Empty input or duplicate clip_id')
    for branch in p.BRANCHES:
        frame[branch]=pd.to_numeric(frame[branch],errors='coerce')
    p.require(np.isfinite(frame[list(p.BRANCHES)].to_numpy()).all(),'Nonfinite scores')
    target=Path(args.output);p.require(not target.exists(),'Refuse to overwrite predictions')
    values=p.probability(bundle,frame)
    result=frame[['clip_id']].assign(generated_probability=values,
        prediction_generated=(values>=bundle['threshold']).astype(int),threshold=bundle['threshold'])
    target.parent.mkdir(parents=True,exist_ok=True);result.to_csv(target,index=False,encoding='utf-8-sig')
    print(f'Predicted {len(result)} videos using frozen {p.digest(Path(cfg["output_dir"])/"stacker.joblib")}')

if __name__=='__main__':main()
