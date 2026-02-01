import WidgetKit
import SwiftUI

struct SlurmEntry: TimelineEntry {
    let date: Date
    let snapshot: ClusterSnapshot?
    let error: String?
    let lastUpdated: Date?
}

struct SlurmProvider: TimelineProvider {
    func placeholder(in context: Context) -> SlurmEntry {
        SlurmEntry(date: Date(), snapshot: nil, error: nil, lastUpdated: nil)
    }

    func getSnapshot(in context: Context, completion: @escaping (SlurmEntry) -> Void) {
        // For previews, just show placeholder
        if context.isPreview {
            completion(SlurmEntry(date: Date(), snapshot: nil, error: nil, lastUpdated: nil))
            return
        }
        // Load cached data
        if let cached = SnapshotCache.load() {
            completion(SlurmEntry(date: Date(), snapshot: cached.snapshot, error: cached.error, lastUpdated: cached.fetchedAt))
        } else {
            completion(SlurmEntry(date: Date(), snapshot: nil, error: nil, lastUpdated: nil))
        }
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<SlurmEntry>) -> Void) {
        // Widget reads from cache only - main app handles SSH fetching
        let refresh = 300 // 5 minutes default
        let next = Date().addingTimeInterval(Double(refresh))

        let entry: SlurmEntry
        if let cached = SnapshotCache.load() {
            entry = SlurmEntry(date: Date(), snapshot: cached.snapshot, error: cached.error, lastUpdated: cached.fetchedAt)
        } else if let errorInfo = SnapshotCache.loadError() {
            entry = SlurmEntry(date: Date(), snapshot: nil, error: errorInfo.error, lastUpdated: errorInfo.fetchedAt)
        } else {
            // No cached data yet - show waiting state
            entry = SlurmEntry(date: Date(), snapshot: nil, error: "Open SlurmHUD app to fetch data", lastUpdated: nil)
        }

        let timeline = Timeline(entries: [entry], policy: .after(next))
        completion(timeline)
    }
}

struct SlurmHUDWidgetView: View {
    var entry: SlurmProvider.Entry

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("SlurmHUD")
                .font(.headline)

            if let snapshot = entry.snapshot {
                Text(snapshot.generated_at ?? "")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                ForEach(Array(snapshot.partitions.prefix(3))) { part in
                    HStack {
                        Text(part.partition ?? "")
                            .font(.caption)
                        Spacer()
                        Text("\(part.nodes?.count ?? 0) nodes")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            } else if let error = entry.error {
                Text(error)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .lineLimit(3)
            } else {
                Text("Waiting for data...")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(8)
    }
}

@main
struct SlurmHUDWidget: Widget {
    let kind: String = "SlurmHUDWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: SlurmProvider()) { entry in
            SlurmHUDWidgetView(entry: entry)
                .containerBackground(.fill.tertiary, for: .widget)
        }
        .configurationDisplayName("SlurmHUD")
        .description("SLURM cluster summary")
        .supportedFamilies([.systemSmall, .systemMedium, .systemLarge])
    }
}
