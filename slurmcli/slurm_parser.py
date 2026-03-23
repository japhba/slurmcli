"""Dedicated parsers for SLURM scontrol output and GRES strings."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def split_gres_entries(gres_value: str) -> List[str]:
    """Comma-split that respects parenthesized groups like gpu:h100:8(S:0,1)."""
    entries: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in gres_value:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            entries.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = ''.join(current).strip()
    if tail:
        entries.append(tail)
    return entries


def extract_gpu_total(gres_value: Optional[str]) -> int:
    if not gres_value:
        return 0
    gres_value = gres_value.strip()
    if not gres_value or gres_value in {"(null)", "N/A", "none"}:
        return 0
    total = 0
    for entry in split_gres_entries(gres_value):
        entry = entry.strip()
        if "gpu" not in entry.lower():
            continue
        entry_core = entry.split('(', 1)[0]
        parts = [part for part in entry_core.split(':') if part]
        for part in reversed(parts):
            m = re.search(r"(\d+)$", part)
            if m:
                total += int(m.group(1))
                break
    return total


def extract_gpu_labels(gres_value: Optional[str]) -> List[str]:
    labels: List[str] = []
    if not gres_value:
        return labels
    seen: set[str] = set()
    for raw_entry in split_gres_entries(gres_value):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "gpu" not in entry.lower():
            continue
        entry_core = entry.split('(', 1)[0]
        parts = [part for part in entry_core.split(':') if part]
        if parts and parts[0].lower() == "gpu":
            parts = parts[1:]
        if not parts:
            continue
        if parts and re.fullmatch(r'\d+', parts[-1]):
            parts = parts[:-1]
        label = ':'.join(parts).strip()
        if not label:
            label = "GPU"
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def parse_scontrol_output(text: str) -> Dict[str, str]:
    """Parse `scontrol show node <name>` output into a flat {Key: Value} dict.

    scontrol emits lines like:
        NodeName=gpu-xd670-30 Arch=x86_64 CoresPerSocket=32
        CPUAlloc=0 CPUTot=128 CPULoad=0.00
    Keys and values are separated by '=' and fields by whitespace, but values
    may contain parentheses with commas (e.g. Gres=gpu:h100:8(S:0,1)).
    """
    fields: Dict[str, str] = {}
    # Flatten to a single line for uniform tokenising
    flat = ' '.join(text.split())
    # Regex: Key= then value which may contain balanced parens
    for m in re.finditer(r'(\w+)=((?:[^()\s]|\([^)]*\))*)', flat):
        fields[m.group(1)] = m.group(2)
    return fields


def build_node_info(fields: Dict[str, str], node_name: str) -> Dict[str, Any]:
    """Build a node info dict from parsed scontrol fields."""
    info: Dict[str, Any] = {
        'node': node_name,
        'memory_gb': None,
        'cpus_total': None,
        'cpus_alloc': None,
        'cpu_arch': None,
        'gpus': '0',
        'gpus_total': 0,
        'gpus_in_use': 0,
        'gpu_labels': [],
        'state': 'unknown',
        'users': [],
        'user_count': 0,
        'jobs': [],
    }

    if 'RealMemory' in fields:
        info['memory_gb'] = int(fields['RealMemory']) / 1024

    if 'CPUTot' in fields:
        info['cpus_total'] = int(fields['CPUTot'])
    if 'CPUAlloc' in fields:
        info['cpus_alloc'] = int(fields['CPUAlloc'])

    info['cpu_arch'] = fields.get('Arch') or fields.get('Architecture')

    gres = fields.get('Gres', '')
    if gres and gres not in {'(null)', 'N/A', 'none'}:
        info['gpus'] = gres
        info['gpus_total'] = extract_gpu_total(gres)
        info['gpu_labels'] = extract_gpu_labels(gres)

    gres_used = fields.get('GresUsed', '')
    if gres_used:
        info['gpus_in_use'] = extract_gpu_total(gres_used)

    if info['gpus_in_use'] == 0 and 'AllocTRES' in fields:
        m = re.search(r'gres/gpu=(\d+)', fields['AllocTRES'])
        if m:
            info['gpus_in_use'] = int(m.group(1))

    if 'State' in fields:
        info['state'] = fields['State'].split('+')[0].lower()

    return info
