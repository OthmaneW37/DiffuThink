"""Run the finite StoryPatch experiment; preserve logs and fail on any failed phase."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume existing optimization phases with unchanged settings')
    args = parser.parse_args()
    os.chdir(ROOT)
    env = os.environ.copy()
    # This flag was already verified for this local gfx1200 ROCm installation.
    # It is process-local; callers can override it with their own configuration.
    env.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
    report = ROOT/'reports'/'storypatch-v2'
    report.mkdir(parents=True, exist_ok=True)
    commands = [
        ('baseline', ['-m','diffuthink.storypatch.evaluate','--model','runs/stories-512/best',
                      '--output','reports/storypatch-v2/baseline.json']),
        ('adapt13m', ['-m','diffuthink.storypatch.train','--source','runs/stories-512/best',
                     '--output','runs/storypatch-13m','--steps','3000','--warmup','100',
                     '--lr','0.00005','--eval-every','500']),
        ('scratch58m', ['-m','diffuthink.storypatch.train','--output','runs/storypatch-58m','--stop-after','2500']),
        ('causal58m', ['-m','diffuthink.storypatch.train','--source','runs/storypatch-58m/best',
                      '--output','runs/storypatch-58m-causal','--steps','3500','--warmup','100',
                      '--span-every','0','--selection-mode','causal','--lr','0.00015']),
        ('edit58m', ['-m','diffuthink.storypatch.train','--source','runs/storypatch-58m-causal/best',
                    '--output','runs/storypatch-58m-edit','--steps','6000','--warmup','200','--lr','0.00005']),
        ('evaluate13m', ['-m','diffuthink.storypatch.evaluate','--model','runs/storypatch-13m/best',
                        '--output','reports/storypatch-v2/adapted13m.json']),
        ('evaluate58m', ['-m','diffuthink.storypatch.evaluate','--model','runs/storypatch-58m-edit/best',
                        '--output','reports/storypatch-v2/scratch58m.json']),
    ]
    status_path = ROOT/'runs'/'storypatch-experiment-status.json'
    status_path.parent.mkdir(exist_ok=True)
    for name, command in commands:
        if args.resume and command[1] == 'diffuthink.storypatch.train':
            output = Path(command[command.index('--output')+1])
            if (output/'last.pt').exists():
                command = command + ['--resume']
        state = {'phase': name, 'status': 'running', 'command': [sys.executable]+command}
        status_path.write_text(json.dumps(state, indent=2))
        print(f'Starting {name}', flush=True)
        with (report/f'{name}.log').open('a' if args.resume else 'w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable]+command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            status_path.write_text(json.dumps({**state, 'status':'failed', 'exit_code':result.returncode}, indent=2))
            raise SystemExit(f'{name} failed; inspect {report/name}.log')
        print(f'Completed {name}', flush=True)
    status_path.write_text(json.dumps({'status':'completed'}, indent=2))


if __name__ == '__main__':
    main()
