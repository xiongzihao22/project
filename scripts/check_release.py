"""Check tracked release paths, source syntax, and accidental private material."""
import ast
import json
import re
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    forbidden = {'.idea', '.plot_dependencies', '__pycache__', 'checkpoints',
                 'outputs', 'results', 'runs', 'progress_logs', 'archives'}
    suffixes = {'.pt', '.pth', '.pyc', '.whl', '.zip', '.rar', '.pem', '.key'}
    private = re.compile(r'BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|connect\.bjb|autodl_cloud_removal|/root/autodl-tmp|[CD]:[/\\]+Users[/\\]+ASUS', re.I)
    paths = [Path(p) for p in tracked if p]
    for path in paths:
        if forbidden.intersection(path.parts) or path.suffix.lower() in suffixes:
            raise ValueError(f'Forbidden release artifact: {path}')
        content = (root / path).read_text(encoding='utf-8')
        if path.as_posix() != 'scripts/check_release.py' and private.search(content):
            raise ValueError(f'Potential private material: {path}')
        if (root / path).stat().st_size > 1_000_000:
            raise ValueError(f'Unexpectedly large source file: {path}')
        if path.suffix == '.py':
            ast.parse(content, filename=str(path))
        elif path.suffix == '.json':
            json.loads(content)
    print(json.dumps({'tracked_files': len(paths), 'status': 'passed'}))


if __name__ == '__main__':
    main()
