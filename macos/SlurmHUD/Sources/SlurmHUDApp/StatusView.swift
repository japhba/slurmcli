import SwiftUI

struct StatusView: View {
    @EnvironmentObject var configStore: ConfigStore
    @EnvironmentObject var fetcher: StatusFetcher

    @State private var timer: Timer?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("SlurmHUD")
                    .font(.headline)
                Spacer()
                Button("Refresh") {
                    fetcher.fetch(config: configStore.config, completion: nil)
                }
                .buttonStyle(.borderedProminent)
            }

            if let snapshot = fetcher.snapshot {
                Text("Updated: \(snapshot.generated_at ?? "unknown")")
                    .font(.caption)
                    .foregroundColor(.secondary)
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(snapshot.partitions) { part in
                            PartitionSummaryView(partition: part)
                        }
                    }
                }
            } else if let error = fetcher.lastError {
                Text("Error: \(error)")
                    .foregroundColor(.red)
            } else {
                Text("Waiting for data...")
            }
        }
        .padding(12)
        .frame(width: 360, height: 420)
        .onAppear {
            fetcher.fetch(config: configStore.config, completion: nil)
            scheduleRefresh()
        }
        .onDisappear {
            timer?.invalidate()
            timer = nil
        }
    }

    private func scheduleRefresh() {
        timer?.invalidate()
        let interval = max(30, configStore.config.refreshSeconds)
        timer = Timer.scheduledTimer(withTimeInterval: TimeInterval(interval), repeats: true) { _ in
            fetcher.fetch(config: configStore.config, completion: nil)
        }
    }
}

struct PartitionSummaryView: View {
    let partition: Partition

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(partition.partition ?? "unknown")
                .font(.subheadline)
                .bold()
            if let configs = partition.configs {
                ForEach(configs.indices, id: \.self) { idx in
                    let cfg = configs[idx]
                    HStack {
                        Text("\(cfg.node_count ?? 0) nodes")
                        Text("\(cfg.mem_gb ?? 0)GB")
                        Text("\(cfg.cpus ?? 0)C")
                        Text("\(cfg.gpus ?? 0)G")
                    }
                    .font(.caption)
                    .foregroundColor(.secondary)
                }
            }
            if let nodes = partition.node_details {
                LazyVGrid(columns: Array(repeating: GridItem(.fixed(80), spacing: 6), count: 4), spacing: 6) {
                    ForEach(nodes) { node in
                        NodeCellView(node: node)
                    }
                }
            }
        }
        .padding(8)
        .background(Color.gray.opacity(0.12))
        .cornerRadius(8)
    }
}

struct NodeCellView: View {
    let node: NodeDetail

    private var stateColor: Color {
        let state = (node.state ?? "").lowercased()
        if state.contains("idle") || state.contains("mix") {
            return Color.green.opacity(0.7)
        }
        if state.contains("drain") || state.contains("down") {
            return Color.red.opacity(0.7)
        }
        return Color.orange.opacity(0.7)
    }

    var body: some View {
        VStack(spacing: 2) {
            Text(node.node ?? "?")
                .font(.caption2)
                .lineLimit(1)
            Text("GPU \(node.gpus_in_use ?? 0)/\(node.gpus_total ?? 0)")
                .font(.caption2)
            Text("CPU \(node.cpus_alloc ?? 0)/\(node.cpus_total ?? 0)")
                .font(.caption2)
            Text(node.gpu_label_text ?? "GPU: n/a")
                .font(.caption2)
                .lineLimit(1)
            Text(HUDHelpers.formatJobSummary(node.jobs))
                .font(.caption2)
                .lineLimit(1)
        }
        .padding(6)
        .frame(width: 80, height: 80)
        .background(stateColor)
        .cornerRadius(6)
    }
}
