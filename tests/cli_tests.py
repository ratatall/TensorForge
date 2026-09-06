#!/usr/bin/env python3
"""CLI contracts checked in fresh temporary directories, with no external packages."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile

cli, root, group, config = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
source = root / 'examples/relu_chain.tf'

def run(args, text, success=True):
    result = subprocess.run([str(cli), *map(str, args)], capture_output=True, text=True)
    output = result.stdout + result.stderr
    if (result.returncode == 0) != success or text not in output:
        raise RuntimeError(f'{args}: exit {result.returncode}: {output}')
    return result

if group == 'options':
    run(['--help'], '--llvm-opt=none|O2')
    run([], 'TensorForge', False)
    run(['bogus', source], 'unknown command', False)
    run(['run', source, '--seed'], 'missing value', False)
    for number in ('-1', '4294967296', '1junk', ''):
        run(['run', source, '--seed', number], 'unsigned 32-bit integer', False)
    for level in ('O3', 'o2', ''):
        run(['run', source, '--llvm-opt=' + level], '--llvm-opt must be none or O2', False)
    run(['check', source, '--llvm-opt=O2'], 'inapplicable option', False)
    run(['dump-ir', source, '--trace-passes'], 'requires --opt', False)
    run(['run', source, '--interpret', '--opt'], 'omit --opt', False)
    run(['run', source, '--interpret', '--llvm-opt=O2'], 'requires --llvm-opt=none', False)
    run(['check', root / 'nonexistent.tf'], 'cannot open', False)
    run(['run', source, '--interpret'], 'result (1024')
    for level in ('none', 'O2'):
        for project in ([], ['--opt']):
            args=['run',source,*project,'--llvm-opt='+level,'--seed','0','--verify']
            run(args,'all 4 JIT configurations')
            run(['emit-llvm',source,*project,'--llvm-opt='+level], '@tensorforge_run')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'nul.tf';path.write_bytes(b'return \x00;')
        result=run(['check',path], 'invalid character', False)
        if '\\x00' not in result.stderr or '^' not in result.stderr:
            raise RuntimeError('NUL diagnostic lost source/caret')
elif group == 'benchmark':
    if config != 'Release':
        run(['benchmark',source,'--iterations','3'], 'require a Release build', False)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'smoke.csv'
            run(['benchmark',source,'--iterations','3','--csv',path], 'all outputs verified')
            with path.open(newline='') as stream: rows=list(csv.DictReader(stream))
            if len(rows)!=5: raise RuntimeError('Expected interpreter and four JIT rows')
            if {(r['project_opt'],r['llvm_opt']) for r in rows[1:]} != {('off','none'),('on','none'),('off','O2'),('on','O2')}:
                raise RuntimeError('Missing optimization combination')
            for row in rows:
                for field in ('median_us','p10_us','p90_us','checksum','compile_us'):
                    if not math.isfinite(float(row[field])): raise RuntimeError('Nonfinite '+field)
                if not 0 < float(row['p10_us']) <= float(row['median_us']) <= float(row['p90_us']):
                    raise RuntimeError('Invalid timing percentiles')
                if row['verified']!='true' or row['samples']!='3' or row['build_type']!='Release':
                    raise RuntimeError('Incorrect benchmark metadata')
            if rows[1]['lowered_loops']!='3' or rows[2]['lowered_loops']!='1' or rows[2]['scratch_bytes']!='0':
                raise RuntimeError('Incorrect lowering structure')
            run(['benchmark',source,'--iterations','0'],'iterations must be',False)
            run(['benchmark',source,'--iterations','3','--csv',Path(tmp)/'absent'/'x.csv'],'cannot write CSV',False)
else:
    raise RuntimeError('Unknown CLI test group')
print(f'CLI {group}: passed ({config})')
