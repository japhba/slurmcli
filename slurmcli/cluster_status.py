from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from slurmcli.slurm_parser import (
    build_node_info,
    extract_gpu_labels,
    extract_gpu_total,
    parse_scontrol_output,
)


# --- Helper parsers for status data ------------------------------------------

def parse_cpu_string(cpu_str: str) -> Optional[Dict[str, int]]:
    try:
        parts = cpu_str.split('/')
        if len(parts) == 4:
            allocated, idle, other, total = map(int, parts)
            return {"allocated": allocated, "idle": idle, "other": other, "total": total}
    except Exception:
        pass
    return None


def parse_memory_string(mem_str: str) -> Optional[float]:
    mem_str = mem_str.strip().rstrip('+')
    try:
        u = mem_str.upper()
        if 'T' in u:
            return float(u.replace('T', '').replace('B', '')) * 1024
        if 'G' in u:
            return float(u.replace('G', '').replace('B', ''))
        if 'M' in u:
            return float(u.replace('M', '').replace('B', '')) / 1024
        return float(mem_str) / 1024
    except Exception:
        return None


def expand_nodelist(nodelist: Optional[str]) -> List[str]:
    if not nodelist or nodelist in {"(null)", "None"}:
        return []
    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", str(nodelist)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.SubprocessError, FileNotFoundError):
        return [nodelist]



# extract_gpu_total and extract_gpu_labels are imported from slurm_parser



def format_gpu_labels(labels: Iterable[str]) -> str:
    seen: set[str] = set()
    ordered: List[str] = []
    for label in labels:
        label_clean = label.strip()
        if not label_clean:
            continue
        key = label_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(label_clean)
    return ", ".join(ordered)


def aggregate_gpu_counts(infos: Iterable[Dict[str, Any]]) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    order: List[str] = []
    for info in infos:
        total = int(info.get('gpus_total') or 0)
        labels = info.get('gpu_labels') or []
        if not labels:
            if total:
                label = "unknown"
                if label not in counts:
                    counts[label] = 0
                    order.append(label)
                counts[label] += total
            continue
        if len(labels) == 1:
            label = labels[0]
            if label not in counts:
                counts[label] = 0
                order.append(label)
            counts[label] += total or 1
            continue
        distributed_total = total if total else len(labels)
        share = max(1, distributed_total // len(labels))
        for label in labels:
            if label not in counts:
                counts[label] = 0
                order.append(label)
            counts[label] += share
    return [(label, counts[label]) for label in order]


def format_gpu_counts(counts: Iterable[Tuple[str, int]]) -> str:
    parts: List[str] = []
    for label, count in counts:
        display_label = "unknown" if label == "unknown" else label
        if count > 0:
            parts.append(f"{count}x {display_label}")
        else:
            parts.append(display_label)
    return ", ".join(parts)


def _extract_field(text: str, key: str) -> Optional[str]:
    m = re.search(rf"{re.escape(key)}=([^\s]+)", text)
    if m:
        return m.group(1)
    return None


def _format_tres_string(tres: str) -> str:
    items: List[str] = []
    for raw in tres.split(','):
        entry = raw.strip()
        if not entry:
            continue
        if '=' not in entry:
            items.append(entry)
            continue
        key, value = entry.split('=', 1)
        key = key.strip()
        value = value.strip()
        if key.startswith('gres/'):
            key = key.split('/', 1)[1]
        items.append(f"{key}={value}")
    return ", ".join(items)


def _summarize_job_resources(alloc_tres: Optional[str], req_tres: Optional[str], req_mem: Optional[str]) -> Optional[str]:
    tres_source = alloc_tres or req_tres
    parts: List[str] = []
    if tres_source:
        parts.append(_format_tres_string(tres_source))
    if req_mem and (not tres_source or 'mem=' not in tres_source):
        parts.append(f"mem={req_mem}")
    if not parts:
        return None
    return "; ".join(parts)


def _sanitize_time_field(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    if not text or text in {"-", "UNLIMITED", "N/A", "Not Set", "NOT_SET", "None"}:
        return None
    return text


def _format_job_time_info(elapsed: Optional[str], remaining: Optional[str]) -> Optional[str]:
    elapsed_clean = _sanitize_time_field(elapsed)
    remaining_clean = _sanitize_time_field(remaining)
    time_bits: List[str] = []
    if elapsed_clean:
        time_bits.append(f"elapsed {elapsed_clean}")
    if remaining_clean:
        time_bits.append(f"left {remaining_clean}")
    if not time_bits:
        return None
    return ", ".join(time_bits)


# --- Jobs & users per node ---------------------------------------------------

JOB_RESOURCE_CACHE: Dict[int, Dict[str, Any]] = {}


def get_job_resource_details(job_id: Optional[int]) -> Dict[str, Any]:
    if job_id is None:
        return {}
    cached = JOB_RESOURCE_CACHE.get(job_id)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            ["scontrol", "show", "job", str(job_id)],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        JOB_RESOURCE_CACHE[job_id] = {}
        return {}

    alloc_tres = _extract_field(output, "AllocTRES")
    req_tres = _extract_field(output, "ReqTRES")
    req_mem = _extract_field(output, "ReqMem")

    summary = _summarize_job_resources(alloc_tres, req_tres, req_mem)
    details = {
        "alloc_tres": alloc_tres,
        "req_tres": req_tres,
        "req_mem": req_mem,
        "summary": summary,
    }
    JOB_RESOURCE_CACHE[job_id] = details
    return details


def get_jobs_by_node() -> Dict[str, List[Dict[str, Any]]]:
    """Return mapping node -> list of jobs {id,user} currently assigned there."""
    try:
        # %i jobid, %u user, %j jobname, %M elapsed, %L time-left, %N nodelist
        result = subprocess.run(
            "squeue --all -h -o '%i|%u|%j|%M|%L|%N'",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}

    by_node: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for line in result.stdout.strip().splitlines():
        if '|' not in line:
            continue
        parts = line.split('|', 5)
        if len(parts) < 6:
            continue
        jid_s, user, job_name, elapsed, time_left, nodelist = parts[:6]
        jid = None
        try:
            jid = int(jid_s)
        except Exception:
            pass
        job_name = job_name.strip() or None
        elapsed_clean = _sanitize_time_field(elapsed)
        time_left_clean = _sanitize_time_field(time_left)
        for node in expand_nodelist(nodelist.strip()):
            by_node[node].append(
                {
                    "id": jid,
                    "user": user,
                    "name": job_name,
                    "elapsed": elapsed_clean,
                    "time_left": time_left_clean,
                }
            )
    return by_node


def get_pending_jobs_by_partition() -> Dict[str, List[Dict[str, Any]]]:
    """Return pending (queued) jobs grouped by partition name."""
    try:
        # %i jobid, %u user, %j name, %P partition, %M elapsed (since submit),
        # %l requested time limit, %D nodes requested, %r reason for waiting.
        result = subprocess.run(
            "squeue --all -h -t PD -o '%i|%u|%j|%P|%M|%l|%D|%r'",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}

    by_part: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for line in result.stdout.strip().splitlines():
        if '|' not in line:
            continue
        parts = line.split('|', 7)
        if len(parts) < 8:
            continue
        jid_s, user, name, part, elapsed, time_limit, nodes_req, reason = parts
        try:
            jid = int(jid_s)
        except Exception:
            jid = None
        by_part[part.strip()].append(
            {
                "id": jid,
                "user": user.strip() or None,
                "name": name.strip() or None,
                "elapsed": _sanitize_time_field(elapsed),
                "time_limit": _sanitize_time_field(time_limit),
                "nodes_requested": nodes_req.strip() or None,
                "reason": reason.strip() or None,
            }
        )
    return by_part


def get_node_user_map() -> Dict[str, set]:
    try:
        result = subprocess.run(
            "squeue --all -h -o '%u|%N'",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}

    node_users: Dict[str, set] = defaultdict(set)
    expansion_cache: Dict[str, List[str]] = {}

    for line in result.stdout.strip().splitlines():
        if '|' not in line:
            continue
        user, nodelist = line.split('|', 1)
        nodelist = nodelist.strip()
        if not nodelist or nodelist in {"(null)", "None"}:
            continue
        if nodelist not in expansion_cache:
            expansion_cache[nodelist] = expand_nodelist(nodelist)
        for node in expansion_cache[nodelist]:
            node_users[node].add(user.strip())

    return node_users


def query_users_for_node(node: str) -> List[str]:
    try:
        result = subprocess.run(
            ["squeue", "--all", "-h", "-w", node, "-o", "%u"],
            check=True,
            capture_output=True,
            text=True,
            timeout=6,
        )
        users = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        return sorted(users)
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def get_detailed_node_info(
    nodes: Iterable[str],
    *,
    include_jobs: bool = False,
    include_job_resources: bool = False,
) -> Dict[str, Dict[str, Any]]:
    node_users_map = get_node_user_map() if include_jobs else {}
    jobs_by_node = get_jobs_by_node() if include_jobs else {}

    node_info: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        try:
            cmd = f"scontrol show node {node}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=8)
            fields = parse_scontrol_output(result.stdout)
            info = build_node_info(fields, node)

            if include_jobs:
                users = sorted(node_users_map.get(node, set()))
                if not users and (info.get('gpus_in_use', 0) or info.get('cpus_alloc', 0)):
                    users = query_users_for_node(node)
                info['users'] = users
                info['user_count'] = len(users)
                job_records = []
                for j in jobs_by_node.get(node, []):
                    jid = j.get("id")
                    resources = get_job_resource_details(jid) if include_job_resources else {}
                    job_records.append(
                        {
                            "id": jid,
                            "user": j.get("user"),
                            "name": j.get("name"),
                            "elapsed": j.get("elapsed"),
                            "time_left": j.get("time_left"),
                            "resources": resources,
                        }
                    )
                info['job_records'] = job_records
                info['jobs'] = job_records

            node_info[node] = info
        except Exception:
            continue

    return node_info


def get_partitions(show_all: bool = False) -> Optional[Dict[int, Dict[str, Any]]]:
    """Parse `sinfo` for partitions and group nodes; optionally include admin/hidden with -a."""
    try:
        base = "sinfo -h -o '%P|%t|%C|%G|%N'"
        cmd = ("sinfo -a -h -o '%P|%t|%C|%G|%N'") if show_all else base
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')

        part_data = defaultdict(lambda: {"states": set(), "cpu_stats": [], "gres": set(), "nodes": set()})
        expansion_cache: Dict[str, List[str]] = {}

        for line in lines:
            parts = line.split('|')
            if len(parts) < 5:
                continue
            partition, state, cpus, gres, nodes = parts[:5]
            partition = partition.strip('*')

            include = show_all or (("idle" in state.lower()) or ("mix" in state.lower()))
            if not include:
                continue

            part_data[partition]['states'].add(state)
            part_data[partition]['cpu_stats'].append(cpus)
            part_data[partition]['gres'].add(gres)

            part_data[partition]['states'].add(state)
            part_data[partition]['cpu_stats'].append(cpus)
            part_data[partition]['gres'].add(gres)

            # Fix: Do not split by comma manually, as nodelists can contain brackets like "node[1-3],other[4]"
            # Passing the entire string to expand_nodelist (via scontrol) handles this correctly.
            if nodes and nodes not in {"(null)", "None"}:
                if nodes not in expansion_cache:
                    expansion_cache[nodes] = expand_nodelist(nodes)
                for expanded in expansion_cache[nodes]:
                    part_data[partition]['nodes'].add(expanded)

        if not part_data:
            return None

        choices: Dict[int, Dict[str, Any]] = {}
        for i, (partition, pdata) in enumerate(part_data.items(), start=1):
            choices[i] = {
                "partition": partition,
                "states": sorted(pdata['states']),
                "cpu_info": parse_cpu_string(pdata['cpu_stats'][0]) if pdata['cpu_stats'] else None,
                "nodes": sorted(pdata['nodes']),
            }
        return choices

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_cluster_snapshot(
    *,
    show_all: bool,
    include_jobs: bool,
    include_job_resources: bool,
) -> Dict[str, Any]:
    parts = get_partitions(show_all=show_all)
    if not parts:
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "partitions": [],
            "show_all": show_all,
            "include_jobs": include_jobs,
            "include_job_resources": include_job_resources,
        }

    all_nodes = sorted({n for p in parts.values() for n in p["nodes"]})
    node_details = get_detailed_node_info(
        all_nodes,
        include_jobs=include_jobs,
        include_job_resources=include_job_resources,
    )
    pending_by_partition = get_pending_jobs_by_partition() if include_jobs else {}

    partitions: List[Dict[str, Any]] = []
    for idx, pdata in sorted(parts.items()):
        nodes = pdata.get("nodes", [])
        configs: Dict[Tuple[int, int, int], List[str]] = defaultdict(list)
        for node in nodes:
            nd = node_details.get(node)
            if nd is None:
                continue
            mem_gb = int((nd.get('memory_gb') or 0))
            cpus = int(nd.get('cpus_total') or 0)
            gpus = int(nd.get('gpus_total') or 0)
            configs[(mem_gb, cpus, gpus)].append(node)

        config_rows: List[Dict[str, Any]] = []
        for (mem_gb, cpus, gpus), bucket_nodes in sorted(
            configs.items(),
            key=lambda x: (x[0][0], x[0][1]),
            reverse=True,
        ):
            infos = [node_details[n] for n in bucket_nodes if n in node_details]
            available_count = sum(1 for i in infos if i.get('state') in {'idle', 'mixed', 'mix'})
            gpu_in_use = sum(int(i.get('gpus_in_use') or 0) for i in infos)
            gpu_total = sum(int(i.get('gpus_total') or 0) for i in infos)
            gpu_counts = aggregate_gpu_counts(infos) if gpus else []
            users_per_node = [len(i.get('users', [])) for i in infos]
            config_rows.append(
                {
                    "mem_gb": mem_gb,
                    "cpus": cpus,
                    "gpus": gpus,
                    "node_count": len(bucket_nodes),
                    "available_count": available_count,
                    "gpu_in_use": gpu_in_use,
                    "gpu_total": gpu_total,
                    "gpu_counts": [{"label": label, "count": count} for label, count in gpu_counts],
                    "users_min": min(users_per_node) if users_per_node else 0,
                    "users_max": max(users_per_node) if users_per_node else 0,
                    "nodes": sorted(bucket_nodes),
                }
            )

        node_rows: List[Dict[str, Any]] = []
        for node in sorted(nodes):
            info = node_details.get(node) or {}
            gpu_label_text = format_gpu_labels(info.get('gpu_labels', []))
            if not gpu_label_text and info.get('gpus_total'):
                gpu_label_text = "unknown"
            job_rows: List[Dict[str, Any]] = []
            if include_jobs:
                for rec in info.get("job_records") or []:
                    name = rec.get("name") or "-"
                    jid = rec.get("id")
                    time_info = _format_job_time_info(rec.get("elapsed"), rec.get("time_left"))
                    resource_summary = (rec.get('resources') or {}).get('summary')
                    job_rows.append(
                        {
                            "id": jid,
                            "user": rec.get("user"),
                            "name": name,
                            "elapsed": rec.get("elapsed"),
                            "time_left": rec.get("time_left"),
                            "time_info": time_info,
                            "resource_summary": resource_summary,
                        }
                    )

            node_rows.append(
                {
                    "node": node,
                    "state": info.get("state") or "unknown",
                    "memory_gb": info.get("memory_gb"),
                    "cpus_alloc": info.get("cpus_alloc"),
                    "cpus_total": info.get("cpus_total"),
                    "cpu_arch": info.get("cpu_arch"),
                    "gpus_in_use": info.get("gpus_in_use"),
                    "gpus_total": info.get("gpus_total"),
                    "gpu_labels": info.get("gpu_labels") or [],
                    "gpu_label_text": gpu_label_text,
                    "users": info.get("users") or [],
                    "user_count": info.get("user_count") or 0,
                    "jobs": job_rows,
                }
            )

        partition_name = pdata.get("partition")
        pending_rows = pending_by_partition.get(partition_name or "", []) if include_jobs else []

        partitions.append(
            {
                "idx": idx,
                "partition": partition_name,
                "states": pdata.get("states") or [],
                "cpu_info": pdata.get("cpu_info"),
                "nodes": nodes,
                "configs": config_rows,
                "node_details": node_rows,
                "pending_jobs": pending_rows,
            }
        )

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "partitions": partitions,
        "show_all": show_all,
        "include_jobs": include_jobs,
        "include_job_resources": include_job_resources,
    }
