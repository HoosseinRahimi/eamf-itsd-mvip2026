"""Execute the unchanged collector for a bounded number of new complete records.

Useful when the local command host interrupts long processes. The collection
algorithm and protocol signature are unchanged; an interruption occurs only after
a complete per-image JSON has been saved. Runs are logged with the wrapper hash.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import collect_revision as collector
from provenance_utils import sha256_file

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--max-new', type=int, default=64)
args, remaining = p.parse_known_args()
output_index = remaining.index('--output') + 1
output = Path(remaining[output_index])
output.mkdir(parents=True, exist_ok=True)
original_write = collector.write_json
count = 0

def bounded_write(path, value):
    global count
    original_write(path, value)
    if isinstance(value, dict) and 'views' in value and 'image' in value:
        count += 1
        if count >= args.max_new:
            log = {'created_utc': datetime.now(timezone.utc).isoformat(),
                   'wrapper_sha256': sha256_file(Path(__file__)),
                   'completed_new_images': count, 'last_image': value['image'],
                   'last_condition': value['condition']}
            with (output / 'bounded_runs.jsonl').open('a', encoding='utf8') as handle:
                handle.write(json.dumps(log) + '\n')
            print(f'BOUNDED_BATCH_COMPLETE {count} new images', flush=True)
            raise SystemExit(0)

collector.write_json = bounded_write
sys.argv = [str(Path(collector.__file__)), *remaining]
collector.main()
