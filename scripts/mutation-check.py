#!/usr/bin/env python3
"""Test regression detection in an isolated copy; never mutate the working sources."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--clean-only", action="store_true", help="Build, test and demo a clean copy without mutations")
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
log_path = root / ('results/clean-check.log' if args.clean_only else 'results/mutation-check.log')
log_path.parent.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='tensorforge-mutation-') as temp, log_path.open('w') as log:
    source = Path(temp) / 'source'
    def ignore(directory, names):
        return [name for name in names if name in {'.git', 'results', '__pycache__', '.DS_Store'}
                or name.endswith('.profraw')
                or (name.startswith('build') and (Path(directory) / name).is_dir())]
    shutil.copytree(root, source, ignore=ignore)
    def run(args, failed=False, contains=None):
        result = subprocess.run([str(a) for a in args], cwd=source, text=True, capture_output=True)
        output = result.stdout + result.stderr
        log.write('$ ' + ' '.join(str(a) for a in args) + '\n' + output + f'exit={result.returncode}\n')
        log.flush()
        if (result.returncode != 0) != failed or (contains and contains not in output):
            raise RuntimeError(f'Unexpected mutation-check result; see {log_path}')
    run(['scripts/build.sh', 'Release', 'build'])
    run(['ctest', '--test-dir', 'build', '-N'])
    run(['ctest', '--test-dir', 'build', '--output-on-failure'])
    run(['scripts/demo.sh'])
    if not args.clean_only:
        run(['build/tf_tests', 'assertion-probe'], failed=True, contains='intentional assertion probe')
    mutations = [
        ('src/ir.cpp', 'symbols_.contains(node.name)', 'false', 'semantic', 'invalid program accepted'),
        ('src/codegen.cpp', 'builder_.CreateFAdd(a, get(op.operands.at(1)), "add")',
         'builder_.CreateFSub(a, get(op.operands.at(1)), "add")', 'differential', 'verification failed'),
    ]
    for filename, before, after, group, message in ([] if args.clean_only else mutations):
        path = source / filename
        original = path.read_text()
        if original.count(before) != 1:
            raise RuntimeError(f'Mutation anchor changed: {filename}')
        try:
            path.write_text(original.replace(before, after))
            run(['cmake', '--build', 'build', '--parallel', '4'])
            run(['ctest', '--test-dir', 'build', '-R', '^' + group + '$', '--output-on-failure'],
                failed=True, contains=message)
            if group == 'differential':
                run(['build/tensorforge', 'run', 'examples/relu_chain.tf', '--verify'],
                    failed=True, contains='verification failed')
        finally:
            path.write_text(original)
        run(['cmake', '--build', 'build', '--parallel', '4'])
        run(['ctest', '--test-dir', 'build', '-R', '^' + group + '$', '--output-on-failure'])
        if path.read_text() != original:
            raise RuntimeError('Mutation was not restored')
print(f'Clean-copy build, tests, and demo passed. Log: {log_path}' if args.clean_only else f'Two mutations detected and restored; assertion and CLI failures propagated. Log: {log_path}')
