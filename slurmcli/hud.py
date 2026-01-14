from __future__ import annotations

import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from slurmcli.cluster_status import build_cluster_snapshot


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _render_html(config: Dict[str, Any]) -> str:
    refresh_ms = int(config.get("refresh_ms") or 5000)
    show_all = "true" if config.get("show_all") else "false"
    include_jobs = "true" if config.get("include_jobs") else "false"
    include_job_resources = "true" if config.get("include_job_resources") else "false"

    html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>SLURM Cluster HUD</title>
    <style>
      :root {
        --bg: #0f172a;
        --panel: #111827;
        --panel-alt: #0b1220;
        --text: #e2e8f0;
        --muted: #94a3b8;
        --accent: #38bdf8;
        --good: #22c55e;
        --warn: #f97316;
        --bad: #ef4444;
        --border: #1f2937;
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        font-family: "Source Code Pro", "Fira Code", "SFMono-Regular", Menlo, Consolas, monospace;
        background: radial-gradient(circle at top left, #0b1020, #0f172a 40%, #05080f 100%);
        color: var(--text);
      }
      header {
        padding: 20px 28px;
        border-bottom: 1px solid var(--border);
        background: rgba(10, 15, 30, 0.8);
        position: sticky;
        top: 0;
        z-index: 10;
        backdrop-filter: blur(6px);
      }
      header h1 {
        margin: 0 0 6px 0;
        font-size: 20px;
        font-weight: 600;
        color: var(--accent);
      }
      header p {
        margin: 0;
        font-size: 12px;
        color: var(--muted);
      }
      .overview {
        margin-top: 16px;
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      }
      .partition-card {
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px;
        background: rgba(15, 23, 42, 0.7);
      }
      .partition-card h3 {
        margin: 0 0 6px 0;
        font-size: 14px;
        color: var(--text);
      }
      .partition-card .stats {
        display: grid;
        gap: 6px;
        font-size: 12px;
        color: var(--muted);
      }
      .node-grid {
        margin-top: 10px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
        gap: 4px;
      }
      .node-cell {
        width: 96px;
        height: 96px;
        border-radius: 3px;
        background: var(--border);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        color: #0b1120;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        padding: 2px;
        text-align: center;
        line-height: 1.2;
      }
      .node-cell.good {
        background: var(--good);
        color: #052e14;
      }
      .node-cell.warn {
        background: var(--warn);
        color: #3f1a00;
      }
      .node-cell.bad {
        background: var(--bad);
        color: #3b0a0a;
      }
      .controls {
        margin-top: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
      }
      .controls label {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        font-size: 12px;
        color: var(--muted);
      }
      .controls input[type="checkbox"] {
        accent-color: var(--accent);
      }
      .controls button {
        padding: 6px 10px;
        font-size: 12px;
        background: transparent;
        border: 1px solid var(--border);
        color: var(--text);
        border-radius: 6px;
        cursor: pointer;
      }
      .controls button:hover {
        border-color: var(--accent);
      }
      main {
        padding: 20px 28px 40px 28px;
      }
      .partition {
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 18px;
        background: linear-gradient(135deg, rgba(17,24,39,0.9), rgba(8,12,24,0.95));
      }
      .partition h2 {
        margin: 0 0 8px 0;
        font-size: 16px;
        color: var(--accent);
      }
      .partition .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 12px;
      }
      .config-list {
        margin: 0 0 16px 0;
        padding: 0;
        list-style: none;
        display: grid;
        gap: 8px;
      }
      .config-list li {
        padding: 10px;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--panel);
        font-size: 12px;
        color: var(--muted);
      }
      .config-list span {
        color: var(--text);
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
      }
      thead th {
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid var(--border);
        color: var(--muted);
      }
      tbody td {
        padding: 8px;
        border-bottom: 1px solid var(--border);
        vertical-align: top;
      }
      tbody tr:hover {
        background: rgba(56, 189, 248, 0.06);
      }
      .state {
        font-weight: 600;
      }
      .state.good {
        color: var(--good);
      }
      .state.warn {
        color: var(--warn);
      }
      .state.bad {
        color: var(--bad);
      }
      .jobs {
        display: grid;
        gap: 4px;
      }
      .job {
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 6px 8px;
        background: var(--panel-alt);
      }
      .job-title {
        color: var(--text);
        font-weight: 600;
      }
      .muted {
        color: var(--muted);
      }
      .empty {
        color: var(--muted);
        font-style: italic;
      }
      @media (max-width: 900px) {
        header {
          padding: 16px;
        }
        main {
          padding: 16px;
        }
        table {
          display: block;
          overflow-x: auto;
        }
      }
    </style>
  </head>
  <body>
    <header>
      <h1>SLURM Cluster HUD</h1>
      <p id="last-updated">Waiting for data...</p>
      <div class="controls">
        <label><input id="toggle-show-all" type="checkbox"/> Show admin partitions</label>
        <label><input id="toggle-jobs" type="checkbox"/> Include jobs</label>
        <label><input id="toggle-job-resources" type="checkbox"/> Include job resources</label>
        <button id="refresh-now">Refresh now</button>
      </div>
      <div id="overview" class="overview"></div>
    </header>
    <main>
      <div id="status"></div>
      <div id="partitions"></div>
    </main>
    <script>
      const refreshMs = %%REFRESH_MS%%;
      const config = {
        showAll: %%SHOW_ALL%%,
        includeJobs: %%INCLUDE_JOBS%%,
        includeJobResources: %%INCLUDE_JOB_RESOURCES%%,
      };

      const statusEl = document.getElementById("status");
      const lastUpdatedEl = document.getElementById("last-updated");
      const partitionsEl = document.getElementById("partitions");
      const overviewEl = document.getElementById("overview");
      const toggleShowAll = document.getElementById("toggle-show-all");
      const toggleJobs = document.getElementById("toggle-jobs");
      const toggleJobResources = document.getElementById("toggle-job-resources");
      const refreshNowBtn = document.getElementById("refresh-now");

      toggleShowAll.checked = config.showAll;
      toggleJobs.checked = config.includeJobs;
      toggleJobResources.checked = config.includeJobResources;

      toggleShowAll.addEventListener("change", () => {
        config.showAll = toggleShowAll.checked;
        fetchStatus();
      });
      toggleJobs.addEventListener("change", () => {
        config.includeJobs = toggleJobs.checked;
        if (!config.includeJobs) {
          config.includeJobResources = false;
          toggleJobResources.checked = false;
        }
        fetchStatus();
      });
      toggleJobResources.addEventListener("change", () => {
        config.includeJobResources = toggleJobResources.checked;
        if (config.includeJobResources) {
          config.includeJobs = true;
          toggleJobs.checked = true;
        }
        fetchStatus();
      });
      refreshNowBtn.addEventListener("click", () => fetchStatus());

      function valueOr(value, fallback) {
        return (value === null || value === undefined) ? fallback : value;
      }

      function stateClass(state) {
        if (!state) return "bad";
        const text = state.toLowerCase();
        if (text.includes("idle") || text.includes("mix")) return "good";
        if (text.includes("drain") || text.includes("down")) return "bad";
        return "warn";
      }

      function render(snapshot) {
        statusEl.innerHTML = "";
        partitionsEl.innerHTML = "";
        overviewEl.innerHTML = "";
        if (!snapshot.partitions || snapshot.partitions.length === 0) {
          statusEl.innerHTML = '<p class="empty">No partition data available.</p>';
          return;
        }
        snapshot.partitions.forEach((part) => {
          const card = document.createElement("div");
          card.className = "partition-card";
          const title = document.createElement("h3");
          title.textContent = part.partition || "unknown";
          card.appendChild(title);

          const stats = document.createElement("div");
          stats.className = "stats";
          const nodeCount = (part.nodes || []).length;
          const available = (part.configs || []).reduce((acc, cfg) => acc + (cfg.available_count || 0), 0);
          const totalGpus = (part.configs || []).reduce((acc, cfg) => acc + (cfg.gpu_total || 0), 0);
          const usedGpus = (part.configs || []).reduce((acc, cfg) => acc + (cfg.gpu_in_use || 0), 0);
          stats.innerHTML =
            `<div>Nodes: <span>${nodeCount}</span></div>` +
            `<div>Available: <span>${available}</span></div>` +
            `<div>GPU usage: <span>${usedGpus}/${totalGpus}</span></div>`;
          card.appendChild(stats);

          const nodeGrid = document.createElement("div");
          nodeGrid.className = "node-grid";
          (part.node_details || []).forEach((node) => {
            const cell = document.createElement("div");
            cell.className = "node-cell " + stateClass(node.state || "");
            const nodeLabel = (node.node || "").split("-").slice(-1)[0] || node.node || "?";
            const gpuSummary = `${valueOr(node.gpus_in_use, 0)}/${valueOr(node.gpus_total, 0)}`;
            const cpuSummary = `${valueOr(node.cpus_alloc, 0)}/${valueOr(node.cpus_total, 0)}`;
            const gpuType = node.gpu_label_text || "";
            const jobCount = node.jobs ? node.jobs.length : 0;
            const jobName = (node.jobs && node.jobs[0] && node.jobs[0].name) ? node.jobs[0].name : "";
            const jobUser = (node.jobs && node.jobs[0] && node.jobs[0].user) ? node.jobs[0].user : "";
            const jobText = jobCount ? `${jobCount} job${jobCount === 1 ? "" : "s"}` : "no jobs";
            cell.innerHTML = `${nodeLabel}<br>${gpuSummary}g ${cpuSummary}c<br>${gpuType}<br>${jobText}`;
            if (jobName || jobUser) {
              const jobDetail = jobUser ? `${jobName} (${jobUser})` : jobName;
              cell.innerHTML += `<br>${jobDetail}`;
            }
            cell.title = `${node.node} (${node.state || "unknown"}) | GPUs ${gpuSummary} | CPUs ${cpuSummary}`;
            nodeGrid.appendChild(cell);
          });
          card.appendChild(nodeGrid);

          overviewEl.appendChild(card);
        });
        snapshot.partitions.forEach((part) => {
          const wrap = document.createElement("section");
          wrap.className = "partition";

          const title = document.createElement("h2");
          title.textContent = "Partition: " + (part.partition || "unknown");
          wrap.appendChild(title);

          const meta = document.createElement("div");
          meta.className = "meta";
          const states = (part.states || []).join(", ");
          if (part.cpu_info) {
            const cpu = part.cpu_info;
            const avail = valueOr(cpu.idle, 0);
            meta.innerHTML += `<div>CPUs: <span>${avail}/${cpu.total}</span> available (${cpu.allocated} in use)</div>`;
          }
          meta.innerHTML += `<div>States: <span>${states || "-"}</span></div>`;
          meta.innerHTML += `<div>Nodes: <span>${(part.nodes || []).length}</span></div>`;
          wrap.appendChild(meta);

          if (part.configs && part.configs.length) {
            const list = document.createElement("ul");
            list.className = "config-list";
            part.configs.forEach((cfg) => {
              const li = document.createElement("li");
              const gpuCounts = (cfg.gpu_counts || []).map((g) => `${g.count}x ${g.label}`).join(", ");
              const users = `${cfg.users_min}-${cfg.users_max}`;
              li.innerHTML =
                `<span>${cfg.node_count} nodes</span> :: ${cfg.mem_gb}GB RAM, ${cfg.cpus} CPUs, ${cfg.gpus} GPUs ` +
                `(available: <span>${cfg.available_count}</span>, GPU usage: <span>${cfg.gpu_in_use}/${cfg.gpu_total}</span>` +
                (gpuCounts ? `, GPU types: <span>${gpuCounts}</span>` : "") +
                `, users/node: <span>${users}</span>)`;
              list.appendChild(li);
            });
            wrap.appendChild(list);
          }

          if (part.node_details && part.node_details.length) {
            const table = document.createElement("table");
            const thead = document.createElement("thead");
            const headRow = document.createElement("tr");
            const headers = ["Node", "State", "Memory", "CPUs", "GPUs", "Users"];
            if (config.includeJobs) headers.push("Jobs");
            headers.forEach((label) => {
              const th = document.createElement("th");
              th.textContent = label;
              headRow.appendChild(th);
            });
            thead.appendChild(headRow);
            table.appendChild(thead);

            const tbody = document.createElement("tbody");
            part.node_details.forEach((node) => {
              const tr = document.createElement("tr");
              const memValue = valueOr(node.memory_gb, null);
              const mem = memValue ? `${memValue.toFixed(0)}GB` : "n/a";
              const cpu = `${valueOr(node.cpus_alloc, 0)}/${valueOr(node.cpus_total, "n/a")}`;
              let gpu = `${valueOr(node.gpus_in_use, 0)}/${valueOr(node.gpus_total, 0)}`;
              if (node.gpu_label_text) {
                gpu += ` (${node.gpu_label_text})`;
              }
              const users = node.user_count ? node.users.join(", ") : "-";
              const cells = [
                node.node,
                node.state || "unknown",
                mem,
                cpu,
                gpu,
                users,
              ];
              cells.forEach((value, idx) => {
                const td = document.createElement("td");
                if (idx === 1) {
                  td.innerHTML = `<span class="state ${stateClass(value)}">${value}</span>`;
                } else {
                  td.textContent = value;
                }
                tr.appendChild(td);
              });
              if (config.includeJobs) {
                const td = document.createElement("td");
                if (node.jobs && node.jobs.length) {
                  const jobsWrap = document.createElement("div");
                  jobsWrap.className = "jobs";
                  node.jobs.forEach((job) => {
                    const jobEl = document.createElement("div");
                    jobEl.className = "job";
                    const title = document.createElement("div");
                    title.className = "job-title";
                    title.textContent = job.name || "-";
                    const meta = document.createElement("div");
                    meta.className = "muted";
                    const bits = [];
                    if (job.user) bits.push(`user ${job.user}`);
                    if (job.time_info) bits.push(job.time_info);
                    if (config.includeJobResources && job.resource_summary) {
                      bits.push(job.resource_summary);
                    }
                    meta.textContent = bits.join(" | ");
                    jobEl.appendChild(title);
                    jobEl.appendChild(meta);
                    jobsWrap.appendChild(jobEl);
                  });
                  td.appendChild(jobsWrap);
                } else {
                  td.innerHTML = '<span class="empty">No jobs</span>';
                }
                tr.appendChild(td);
              }
              tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            wrap.appendChild(table);
          }

          partitionsEl.appendChild(wrap);
        });
      }

      function fetchStatus() {
        const params = new URLSearchParams();
        params.set("show_all", config.showAll ? "1" : "0");
        params.set("include_jobs", config.includeJobs ? "1" : "0");
        params.set("include_job_resources", config.includeJobResources ? "1" : "0");
        fetch(`/api/status?${params.toString()}`)
          .then((res) => res.json())
          .then((data) => {
            const timestamp = data.generated_at ? new Date(data.generated_at) : null;
            if (timestamp) {
              lastUpdatedEl.textContent = `Last updated: ${timestamp.toLocaleString()}`;
            }
            render(data);
          })
          .catch((err) => {
            statusEl.innerHTML = `<p class="empty">Failed to load status: ${err}</p>`;
          });
      }

      fetchStatus();
      setInterval(fetchStatus, refreshMs);
    </script>
  </body>
</html>
"""

    return (
        html.replace("%%REFRESH_MS%%", str(refresh_ms))
        .replace("%%SHOW_ALL%%", show_all)
        .replace("%%INCLUDE_JOBS%%", include_jobs)
        .replace("%%INCLUDE_JOB_RESOURCES%%", include_job_resources)
    )


def _make_handler(config: Dict[str, Any]):
    class HudHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_render_html(config))
                return
            if parsed.path == "/api/status":
                params = parse_qs(parsed.query)
                show_all = _parse_bool(params.get("show_all", [None])[0], config.get("show_all", False))
                include_jobs = _parse_bool(params.get("include_jobs", [None])[0], config.get("include_jobs", False))
                include_job_resources = _parse_bool(
                    params.get("include_job_resources", [None])[0],
                    config.get("include_job_resources", False),
                )
                if include_job_resources and not include_jobs:
                    include_jobs = True
                snapshot = build_cluster_snapshot(
                    show_all=show_all,
                    include_jobs=include_jobs,
                    include_job_resources=include_job_resources,
                )
                self._send_json(snapshot)
                return
            if parsed.path == "/api/health":
                self._send_json({"status": "ok", "time": time.time()})
                return
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    return HudHandler


def run_hud(
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
    refresh_seconds: int = 600,
    verbosity: int = 0,
) -> None:
    show_all = verbosity >= 2
    include_jobs = verbosity >= 2
    include_job_resources = verbosity >= 4
    config = {
        "show_all": show_all,
        "include_jobs": include_jobs,
        "include_job_resources": include_job_resources,
        "refresh_ms": int(refresh_seconds * 1000),
    }
    handler = _make_handler(config)
    server = HTTPServer((host, int(port)), handler)
    display_host = "localhost" if host == "0.0.0.0" else host
    print(f"HUD running at http://{display_host}:{port} (refresh {refresh_seconds}s)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping HUD...")
    finally:
        server.server_close()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SLURM cluster usage HUD")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the HUD server")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind the HUD server")
    parser.add_argument("--refresh", type=int, default=600, help="Refresh interval in seconds")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv, -vvv, -vvvv)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_hud(
        host=args.host,
        port=args.port,
        refresh_seconds=args.refresh,
        verbosity=args.verbose,
    )


if __name__ == "__main__":
    main()
