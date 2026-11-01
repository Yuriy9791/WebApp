"""Verify the application read path. This command never writes to S3."""
import argparse
import json
from .data import Repository

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--all',action='store_true',help='Read every Parquet')
    args=p.parse_args(); repo=Repository.from_env(); snap=repo.active(); records=list(snap.records)
    if not args.all:
        records=[k for k,r in snap.records.items() if r['las_filename']=='1_Craigie, GeoArabia.las'] or records[:1]
    samples=0
    for i,rid in enumerate(records,1):
        samples+=repo.table(snap,rid).num_rows
        if args.all and i%20==0: print(f'Checked {i}/{len(records)} Parquet',flush=True)
    print(json.dumps({'status':'OK','release':snap.release,'records':len(snap.records),'plot_curves':len(snap.curves),'parquet_checked':len(records),'samples_checked':samples,'manifest_sha256':snap.digest},indent=2))
if __name__=='__main__': main()
