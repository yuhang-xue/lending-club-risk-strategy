"""Small IO helpers; never imports or modifies historical implementation."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
FEATURES = ('fico_n', 'purpose', 'emp_length', 'home_ownership_n')
MISSING, OTHER, UNKNOWN = '__MISSING__', '__OTHER__', '__UNKNOWN__'
STRATEGIES = ('S0', 'S1', 'SC_WOE')
PLAN = ROOT / '执行方案/四字段WOE评分卡对照实验执行方案.md'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def read(path, **kwargs):
    return pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False,
                       float_precision='round_trip', **kwargs)


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    relative = str(relative).replace('\\', '/')
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'path outside declared root: ' + relative)
    return path
