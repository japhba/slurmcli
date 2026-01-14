import WidgetKit
import SwiftUI

struct SlurmEntry: TimelineEntry {
    let date: Date
    let snapshot: ClusterSnapshot?
    let error: String?
}

struct SlurmProvider: TimelineProvider {
    func placeholder(in context: Context) -> SlurmEntry {
        SlurmEntry(date: Date(), snapshot: nil, error: nil)
    }

    func getSnapshot(in context: Context, completion: @escaping (SlurmEntry) -> Void) {
        let entry = SlurmEntry(date: Date(), snapshot: nil, error: nil)
        completion(entry)
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<SlurmEntry>) -> Void) {
        let configStore = ConfigStore()
        let fetcher = StatusFetcher()
        fetcher.fetch(config: configStore.config) { snapshot in
            let entry = SlurmEntry(date: Date(), snapshot: snapshot, error: fetcher.lastError)
            let refresh = max(300, configStore.config.refreshSeconds)
            let next = Calendar.current.date(byAdding: .second, value: refresh, to: Date()) ?? Date().addingTimeInterval(Double(refresh))
            let timeline = Timeline(entries: [entry], policy: .after(next))
            completion(timeline)
        }
    }
}

struct SlurmHUDWidgetView: View {
    var entry: SlurmProvider.Entry

    var body: some View {
        if let snapshot = entry.snapshot {
            VStack(alignment: .leading, spacing: 6) {
                Text("SlurmHUD")
                    .font(.headline)
                Text(snapshot.generated_at ?? "unknown")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                ForEach(snapshot.partitions.prefix(3)) { part in
                    HStack {
                        Text(part.partition ?? "unknown")
                            .font(.caption)
                        Spacer()
                        Text("\(part.nodes?.count ?? 0) nodes")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding(8)
        } else if let error = entry.error {
            VStack(alignment: .leading) {
                Text("SlurmHUD")
                    .font(.headline)
                Text("Error")
                    .font(.caption)
                Text(error)
                    .font(.caption2)
            }
            .padding(8)
        } else {
            VStack(alignment: .leading) {
                Text("SlurmHUD")
                    .font(.headline)
                Text("Waiting for data...")
                    .font(.caption2)
            }
            .padding(8)
        }
    }
}

@main
struct SlurmHUDWidget: Widget {
    let kind: String = "SlurmHUDWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: SlurmProvider()) { entry in
            SlurmHUDWidgetView(entry: entry)
        }
        .configurationDisplayName("SlurmHUD")
        .description("SLURM cluster summary")
        .supportedFamilies([.systemSmall, .systemMedium, .systemLarge])
    }
}
