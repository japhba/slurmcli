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
            Text("Error: \(error)")
                .foregroundColor(.red)
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
                        if let timestamp = snapshot.generated_at {
                            Text("Updated: \(timestamp)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        if let nodes = partition.node_details, !nodes.isEmpty {
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 240), spacing: 10)], spacing: 10) {
                                ForEach(nodes) { node in
                                    NodeCellView(node: node)
                                }
                            }
                        } else {
                            Text("No node details available.")
                                .foregroundColor(.secondary)
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

    // MARK: - Timer

    private func scheduleRefresh() {
        timer?.invalidate()
        let interval = max(30, configStore.config.refreshSeconds)
        timer = Timer.scheduledTimer(withTimeInterval: TimeInterval(interval), repeats: true) { _ in
            fetcher.fetch(config: configStore.config, completion: nil)
        }
    }
}

struct NodeCellView: View {
    let node: NodeDetail

    private var stateColor: Color {
        let state = (node.state ?? "").lowercased()
        if state.contains("idle") {
            return Color.green.opacity(0.25)
        }
        if state.contains("mix") {
            return Color.green.opacity(0.15)
        }
        if state.contains("drain") || state.contains("down") {
            return Color.red.opacity(0.2)
        }
        return Color.orange.opacity(0.2)
    }

    private var stateBorderColor: Color {
        let state = (node.state ?? "").lowercased()
        if state.contains("idle") { return .green }
        if state.contains("mix") { return .green.opacity(0.6) }
        if state.contains("drain") || state.contains("down") { return .red }
        return .orange
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Header: node name + state
            HStack {
                Text(node.node ?? "?")
                    .font(.system(.body, design: .monospaced))
                    .fontWeight(.bold)
                    .lineLimit(1)
                    .textSelection(.enabled)
                Spacer()
                Text(node.state ?? "?")
                    .font(.caption2)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 2)
                    .background(stateBorderColor.opacity(0.3))
                    .cornerRadius(4)
            }

            Divider()

            // Resource summary
            HStack(spacing: 12) {
                Label("\(node.cpus_alloc ?? 0)/\(node.cpus_total ?? 0)", systemImage: "cpu")
                Label("\(node.gpus_in_use ?? 0)/\(node.gpus_total ?? 0)", systemImage: "rectangle.stack")
                if let mem = node.memory_gb {
                    Label("\(Int(mem))GB", systemImage: "memorychip")
                }
            }
            .font(.caption)
            .foregroundColor(.secondary)

            if let gpuLabel = node.gpu_label_text, !gpuLabel.isEmpty {
                Text(gpuLabel)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
            }

            // Jobs list
            if let jobs = node.jobs, !jobs.isEmpty {
                Divider()
                ForEach(jobs) { job in
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
        .frame(minWidth: 240)
        .background(stateColor)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(stateBorderColor.opacity(0.5), lineWidth: 1)
        )
        .cornerRadius(8)
    }
}
