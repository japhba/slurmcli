import SwiftUI

struct StatusView: View {
    @EnvironmentObject var configStore: ConfigStore
    @EnvironmentObject var fetcher: StatusFetcher

    @State private var timer: Timer?
    @State private var selectedPartitionID: String?
    @State private var showLog = false

    private static let logTimeFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detail
        }
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Button {
                    fetcher.fetch(config: configStore.config, completion: nil)
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(fetcher.isFetching)
            }
            ToolbarItem(placement: .automatic) {
                Button {
                    showLog.toggle()
                } label: {
                    Label("Log", systemImage: "list.bullet.rectangle")
                }
            }
            ToolbarItem(placement: .automatic) {
                SettingsLink {
                    Label("Settings", systemImage: "gear")
                }
            }
        }
        .onAppear {
            fetcher.fetch(config: configStore.config, completion: nil)
            scheduleRefresh()
        }
        .onReceive(configStore.$config) { _ in
            scheduleRefresh()
        }
        .onDisappear {
            timer?.invalidate()
            timer = nil
        }
    }

    // MARK: - Sidebar

    @ViewBuilder
    private var sidebar: some View {
        if let snapshot = fetcher.snapshot {
            List(snapshot.partitions, selection: $selectedPartitionID) { part in
                NavigationLink(value: part.id) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(part.partition ?? "unknown")
                            .font(.headline)
                        if let configs = part.configs {
                            ForEach(configs.indices, id: \.self) { idx in
                                let cfg = configs[idx]
                                Text("\(cfg.node_count ?? 0) nodes  \(cfg.mem_gb ?? 0)GB  \(cfg.cpus ?? 0)C  \(cfg.gpus ?? 0)G")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
            .navigationTitle("Partitions")
        } else if let error = fetcher.lastError {
            Text(isWaitingMessage(error) ? error : "Error: \(error)")
                .foregroundColor(isWaitingMessage(error) ? .secondary : .red)
                .padding()
        } else {
            Text("Waiting for data...")
                .padding()
        }
    }

    // MARK: - Detail

    @ViewBuilder
    private var detail: some View {
        VStack(spacing: 0) {
            if let snapshot = fetcher.snapshot,
               let partID = selectedPartitionID,
               let partition = snapshot.partitions.first(where: { $0.id == partID }) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        if fetcher.isFetching {
                            HStack(spacing: 8) {
                                ProgressView()
                                    .controlSize(.small)
                                Text("Refreshing cluster data…")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        if let timestamp = snapshot.generated_at {
                            Text("Updated: \(Self.formatTimestamp(timestamp))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        if let nodes = partition.node_details, !nodes.isEmpty {
                            LazyVGrid(
                                columns: [GridItem(.adaptive(minimum: 240), spacing: 10, alignment: .top)],
                                alignment: .leading,
                                spacing: 10
                            ) {
                                ForEach(nodes) { node in
                                    NodeCellView(node: node)
                                }
                            }
                        } else {
                            Text("No node details available.")
                                .foregroundColor(.secondary)
                        }
                        if let pending = partition.pending_jobs, !pending.isEmpty {
                            PendingJobsSection(jobs: pending)
                        }
                    }
                    .padding()
                }
                .navigationTitle(partition.partition ?? "Partition")
            } else {
                Text("Select a partition")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            if showLog {
                Divider()
                logPanel
            }
        }
    }

    @ViewBuilder
    private var logPanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Log")
                    .font(.caption)
                    .fontWeight(.semibold)
                Spacer()
                Button("Clear") {
                    fetcher.log.removeAll()
                }
                .font(.caption)
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)

            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(fetcher.log) { entry in
                            Text("\(Self.logTimeFmt.string(from: entry.date))  \(entry.message)")
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(entry.message.hasPrefix("Error") ? .red : .primary)
                                .id(entry.id)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                }
                .onChange(of: fetcher.log.count) { _ in
                    if let last = fetcher.log.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
        .frame(height: 120)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: - Formatting

    private static let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private static let displayFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .medium
        return f
    }()

    private static func formatTimestamp(_ raw: String) -> String {
        guard let date = isoFormatter.date(from: raw) else { return raw }
        let local = displayFormatter.string(from: date)
        let ago = RelativeDateTimeFormatter()
        ago.unitsStyle = .abbreviated
        let relative = ago.localizedString(for: date, relativeTo: Date())
        return "\(local) (\(relative))"
    }

    // MARK: - Timer

    private func scheduleRefresh() {
        timer?.invalidate()
        let interval = max(30, configStore.config.refreshSeconds)
        timer = Timer.scheduledTimer(withTimeInterval: TimeInterval(interval), repeats: true) { _ in
            fetcher.fetch(config: configStore.config, completion: nil)
        }
    }

    private func isWaitingMessage(_ message: String) -> Bool {
        message.localizedCaseInsensitiveContains("next scheduled refresh")
    }
}

struct NodeCellView: View {
    let node: NodeDetail
    @EnvironmentObject var watchStore: WatchStore

    private var isWatched: Bool {
        guard let name = node.node else { return false }
        return watchStore.isWatched(name)
    }

    private var stateLower: String {
        (node.state ?? "").lowercased()
    }

    private var isProblematic: Bool {
        stateLower.contains("drain") || stateLower.contains("down")
    }

    /// 0.0 = fully idle, 1.0 = fully allocated. Uses the max of CPU and GPU
    /// utilization so a node with 1 GPU pinned still reads as busy.
    private var loadFraction: Double {
        if isProblematic { return 1.0 }

        var fractions: [Double] = []
        if let total = node.cpus_total, total > 0 {
            fractions.append(Double(node.cpus_alloc ?? 0) / Double(total))
        }
        if let total = node.gpus_total, total > 0 {
            fractions.append(Double(node.gpus_in_use ?? 0) / Double(total))
        }

        if let observed = fractions.max() {
            return min(1.0, max(0.0, observed))
        }

        // Fall back to the textual state when we have no quantitative load.
        if stateLower.contains("idle") { return 0.0 }
        if stateLower.contains("mix") { return 0.5 }
        if stateLower.contains("alloc") { return 1.0 }
        return 0.5
    }

    /// Hue ramp: 120° (green) → 60° (yellow) → 30° (orange).
    private func loadHue(saturation: Double, brightness: Double, opacity: Double) -> Color {
        let t = loadFraction
        let hueDegrees: Double
        if t <= 0.5 {
            hueDegrees = 120 - 60 * (t / 0.5)            // 120 → 60
        } else {
            hueDegrees = 60 - 30 * ((t - 0.5) / 0.5)     // 60  → 30
        }
        return Color(hue: hueDegrees / 360.0, saturation: saturation, brightness: brightness)
            .opacity(opacity)
    }

    private var stateColor: Color {
        if isProblematic { return Color.red.opacity(0.20) }
        return loadHue(saturation: 0.55, brightness: 0.95, opacity: 0.22)
    }

    private var stateBorderColor: Color {
        if isProblematic { return Color.red }
        return loadHue(saturation: 0.70, brightness: 0.85, opacity: 0.85)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Header: node name + watch + state
            HStack {
                Text(node.node ?? "?")
                    .font(.system(.body, design: .monospaced))
                    .fontWeight(.bold)
                    .lineLimit(1)
                    .textSelection(.enabled)
                Spacer()
                Button {
                    if let name = node.node {
                        watchStore.toggle(name)
                    }
                } label: {
                    Image(systemName: isWatched ? "bell.fill" : "bell")
                        .foregroundColor(isWatched ? .yellow : .secondary)
                }
                .buttonStyle(.plain)
                .help(isWatched ? "Stop watching this node" : "Watch for job completions")
                Text(node.state ?? "?")
                    .font(.caption2)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 2)
                    .background(stateBorderColor.opacity(0.3))
                    .cornerRadius(4)
            }

            Divider()

            // Resource summary — explicit CPU / GPU / MEM labels
            HStack(spacing: 12) {
                ResourceTag(
                    icon: "cpu",
                    label: "CPU",
                    value: "\(node.cpus_alloc ?? 0)/\(node.cpus_total ?? 0)"
                )
                ResourceTag(
                    icon: "cpu.fill",
                    label: "GPU",
                    value: "\(node.gpus_in_use ?? 0)/\(node.gpus_total ?? 0)"
                )
                if let mem = node.memory_gb {
                    ResourceTag(
                        icon: "memorychip",
                        label: "MEM",
                        value: "\(Int(mem))GB"
                    )
                }
                if let disk = node.tmp_disk_gb, disk > 0 {
                    ResourceTag(
                        icon: "internaldrive",
                        label: "DISK",
                        value: formatDiskSize(disk)
                    )
                }
            }
            .font(.caption)
            .foregroundColor(.secondary)

            if let gpuLabel = node.gpu_label_text, !gpuLabel.isEmpty {
                Text(gpuLabel)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                    .truncationMode(.tail)
                    .textSelection(.enabled)
            }

            // Jobs list
            if let jobs = node.jobs, !jobs.isEmpty {
                Divider()
                ForEach(jobs, id: \.jobIdentifier) { job in
                    VStack(alignment: .leading, spacing: 2) {
                        HStack {
                            Text(job.name ?? "unnamed")
                                .font(.caption)
                                .fontWeight(.medium)
                                .lineLimit(2)
                            Spacer()
                            if let user = job.user {
                                Text(user)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                        HStack(spacing: 8) {
                            if let elapsed = job.elapsed {
                                Label(elapsed, systemImage: "clock")
                            }
                            if let left = job.time_left {
                                Label(left, systemImage: "clock.badge.checkmark")
                            }
                        }
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 2)
                }
            } else {
                Text("No jobs")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(10)
        .frame(minWidth: 240, maxWidth: .infinity, alignment: .topLeading)
        .background(stateColor)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(stateBorderColor.opacity(0.5), lineWidth: 1)
        )
        .cornerRadius(8)
    }
}

private func formatDiskSize(_ gb: Double) -> String {
    if gb >= 1024 {
        let tb = gb / 1024
        return String(format: tb >= 10 ? "%.0fTB" : "%.1fTB", tb)
    }
    return "\(Int(gb))GB"
}

private struct ResourceTag: View {
    let icon: String
    let label: String
    let value: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
            Text(label)
                .fontWeight(.semibold)
            Text(value)
        }
    }
}

struct PendingJobsSection: View {
    let jobs: [PendingJob]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "hourglass")
                Text("Queued jobs")
                    .font(.headline)
                Text("(\(jobs.count))")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .padding(.top, 8)

            LazyVGrid(
                columns: [GridItem(.adaptive(minimum: 240), spacing: 8, alignment: .top)],
                alignment: .leading,
                spacing: 8
            ) {
                ForEach(jobs, id: \.jobIdentifier) { job in
                    PendingJobCell(job: job)
                }
            }
        }
    }
}

private struct PendingJobCell: View {
    let job: PendingJob

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(job.name ?? "unnamed")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .lineLimit(2)
                Spacer()
                if let user = job.user {
                    Text(user)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            HStack(spacing: 8) {
                if let elapsed = job.elapsed {
                    Label(elapsed, systemImage: "clock")
                }
                if let limit = job.time_limit, limit != "UNLIMITED" {
                    Label(limit, systemImage: "timer")
                }
                if let n = job.nodes_requested, n != "0" {
                    Label("\(n) node\(n == "1" ? "" : "s")", systemImage: "server.rack")
                }
            }
            .font(.caption2)
            .foregroundColor(.secondary)
            if let reason = job.reason, !reason.isEmpty, reason != "None" {
                Text("Waiting: \(reason)")
                    .font(.caption2)
                    .foregroundColor(.orange)
                    .lineLimit(2)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.yellow.opacity(0.10))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.yellow.opacity(0.45), lineWidth: 1)
        )
        .cornerRadius(6)
    }
}
