import SwiftUI
import WidgetKit

struct SlurmEntry: TimelineEntry {
    let date: Date
    let snapshot: ClusterSnapshot?
    let error: String?
    let lastUpdated: Date?
    let widgetPartition: String?
}

struct SlurmProvider: TimelineProvider {
    func placeholder(in context: Context) -> SlurmEntry {
        SlurmEntry(
            date: Date(),
            snapshot: MockClusterData.snapshot(referenceDate: Date().addingTimeInterval(-120)),
            error: nil,
            lastUpdated: Date().addingTimeInterval(-120),
            widgetPartition: nil
        )
    }

    func getSnapshot(in context: Context, completion: @escaping (SlurmEntry) -> Void) {
        if context.isPreview {
            completion(placeholder(in: context))
            return
        }

        completion(loadEntry())
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<SlurmEntry>) -> Void) {
        let entry = loadEntry()
        let config = ConfigStore.loadShared() ?? .default
        let cadence = max(60, config.refreshSeconds)
        let scheduledRefresh = (entry.lastUpdated ?? entry.date).addingTimeInterval(Double(cadence))
        let nextRefresh =
            scheduledRefresh > Date()
            ? scheduledRefresh
            : Date().addingTimeInterval(min(Double(cadence), 60))
        completion(Timeline(entries: [entry], policy: .after(nextRefresh)))
    }

    private func loadEntry() -> SlurmEntry {
        let now = Date()
        let widgetPartition = ConfigStore.loadShared()?.widgetPartition

        if let cached = SnapshotCache.load() {
            return SlurmEntry(
                date: now,
                snapshot: cached.snapshot,
                error: cached.error,
                lastUpdated: cached.fetchedAt,
                widgetPartition: widgetPartition
            )
        }

        if let errorInfo = SnapshotCache.loadError() {
            return SlurmEntry(
                date: now,
                snapshot: nil,
                error: errorInfo.error,
                lastUpdated: errorInfo.fetchedAt,
                widgetPartition: widgetPartition
            )
        }

        if let accessError = SnapshotCache.sharedAccessErrorMessage() {
            return SlurmEntry(
                date: now,
                snapshot: nil,
                error: accessError,
                lastUpdated: nil,
                widgetPartition: widgetPartition
            )
        }

        return SlurmEntry(
            date: now,
            snapshot: nil,
            error: "Open SlurmHUD to fetch cluster data.",
            lastUpdated: nil,
            widgetPartition: widgetPartition
        )
    }
}

private func normalizePartitionFilter(_ raw: String?) -> String? {
    guard let raw else { return nil }
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }

    return trimmed
        .lowercased()
        .replacingOccurrences(of: " ", with: "_")
}

private func partitionMatchesFilter(_ partitionName: String, filter: String?) -> Bool {
    guard let filter else { return true }
    return normalizePartitionFilter(partitionName) == filter
}

private struct JobResourceSummary {
    let gres: String?
    let cpus: String?
    let memory: String?

    init(raw: String?) {
        var gres: String?
        var cpus: String?
        var memory: String?

        let tokens = (raw ?? "")
            .split(whereSeparator: { $0 == "," || $0 == ";" })
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }

        for token in tokens {
            let parts = token.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { continue }

            let key = parts[0].lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            let value = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)

            if key == "cpu" || key == "cpus" {
                cpus = "c\(value)"
            } else if key == "mem" || key == "memory" {
                memory = "m\(value)"
            } else if key.hasPrefix("gpu") || key.hasPrefix("gres") {
                gres = "g\(value)"
            }
        }

        self.gres = gres
        self.cpus = cpus
        self.memory = memory
    }

    var line: String {
        let parts = [gres, cpus, memory].compactMap { $0 }
        return parts.isEmpty ? "-" : parts.joined(separator: " ")
    }

    var compactLine: String {
        let parts = [gres, cpus, memory]
            .compactMap { compactToken($0) }
        return parts.prefix(2).joined(separator: " ")
    }

    private func compactToken(_ raw: String?) -> String? {
        guard let raw, !raw.isEmpty else { return nil }

        let cleaned = raw.filter { $0.isLetter || $0.isNumber }
        guard let prefix = cleaned.first else { return nil }
        let rest = String(cleaned.dropFirst()).lowercased()

        if prefix == "m" || prefix == "M" {
            return "\(String(prefix).lowercased())\(rest.prefix(4))"
        }

        return "\(String(prefix).lowercased())\(rest.prefix(3))"
    }
}

private struct ClusterWidgetBoard {
    struct JobTile: Identifiable {
        let id: String
        let name: String
        let userTag: String
        let resourceLine: String
        let compactResourceLine: String
        let runtimeTag: String?
    }

    struct NodeTile: Identifiable {
        enum Tone {
            case available
            case mixed
            case busy
            case warning
            case neutral
        }

        let id: String
        let name: String
        let gpuLabel: String
        let stateLabel: String
        let tone: Tone
        let jobs: [JobTile]
    }

    struct PartitionSection: Identifiable {
        let id: String
        let name: String
        let availableNodes: Int
        let totalNodes: Int
        let jobCount: Int
        let nodes: [NodeTile]

        var statusText: String {
            let freeText = totalNodes > 0 ? "\(availableNodes)/\(totalNodes)" : "0/0"
            return jobCount > 0 ? "\(freeText)·\(jobCount)j" : freeText
        }
    }

    let refreshedAt: Date?
    let partitions: [PartitionSection]

    var totalNodes: Int {
        partitions.reduce(0) { $0 + $1.totalNodes }
    }

    var totalJobs: Int {
        partitions.reduce(0) { $0 + $1.jobCount }
    }

    init(snapshot: ClusterSnapshot, lastUpdated: Date?, partitionFilter: String?) {
        self.refreshedAt = parseWidgetTimestamp(snapshot.generated_at) ?? lastUpdated
        self.partitions = snapshot.partitions
            .compactMap(Self.makePartitionSection)
            .filter { partitionMatchesFilter($0.name, filter: partitionFilter) }
    }

    private static func makePartitionSection(from partition: Partition) -> PartitionSection? {
        let configs = partition.configs ?? []
        let details = partition.node_details ?? []
        let nodeNames = partition.nodes ?? []

        let totalFromConfigs = configs.reduce(0) { $0 + ($1.node_count ?? 0) }
        let availableFromConfigs = configs.reduce(0) { $0 + ($1.available_count ?? 0) }
        let availableFromDetails = details.filter { isAvailableState($0.state) }.count
        let totalNodes = max(totalFromConfigs, details.count, nodeNames.count)
        let availableNodes = min(totalNodes, max(availableFromConfigs, availableFromDetails))

        let nodeTiles: [NodeTile]
        if details.isEmpty {
            nodeTiles = nodeNames.map { placeholderNode(named: $0) }
        } else {
            nodeTiles = details
                .map(makeNodeTile)
                .sorted { lhs, rhs in
                    if lhs.jobs.count == rhs.jobs.count {
                        return lhs.name < rhs.name
                    }
                    return lhs.jobs.count > rhs.jobs.count
                }
        }

        let partitionName = partition.partition ?? "Partition"
        let jobCount = nodeTiles.reduce(0) { $0 + $1.jobs.count }

        return PartitionSection(
            id: partition.id,
            name: partitionName,
            availableNodes: availableNodes,
            totalNodes: totalNodes,
            jobCount: jobCount,
            nodes: nodeTiles
        )
    }

    private static func makeNodeTile(from node: NodeDetail) -> NodeTile {
        let resources = (node.jobs ?? []).map { job -> JobTile in
            let parsed = JobResourceSummary(raw: job.resource_summary)
            let name = (job.name?.isEmpty == false ? job.name : nil) ?? "job"
            let user = (job.user?.isEmpty == false ? job.user : nil) ?? "unknown"
            let userTag = String(user.prefix(2))
            let jobID = job.id.map(String.init) ?? "\(name)-\(userTag)"

            return JobTile(
                id: jobID,
                name: name,
                userTag: userTag,
                resourceLine: parsed.line,
                compactResourceLine: parsed.compactLine,
                runtimeTag: compactRuntimeTag(job.elapsed)
            )
        }

        let gpuLabel = (node.gpu_label_text?.isEmpty == false ? node.gpu_label_text : nil) ?? "CPU"
        let state = simplifiedState(node.state)

        return NodeTile(
            id: node.id,
            name: node.node ?? "node",
            gpuLabel: gpuLabel,
            stateLabel: state.label,
            tone: state.tone,
            jobs: resources
        )
    }

    private static func placeholderNode(named name: String) -> NodeTile {
        NodeTile(
            id: name,
            name: name,
            gpuLabel: "Awaiting details",
            stateLabel: "unknown",
            tone: .neutral,
            jobs: []
        )
    }

    private static func isAvailableState(_ raw: String?) -> Bool {
        let state = (raw ?? "").lowercased()
        return state.contains("idle") || state.contains("mix")
    }

    private static func simplifiedState(_ raw: String?) -> (label: String, tone: NodeTile.Tone) {
        let state = (raw ?? "").lowercased()

        if state.contains("drain") || state.contains("down") {
            return ("D", .warning)
        }
        if state.contains("alloc") {
            return ("B", .busy)
        }
        if state.contains("mix") {
            return ("M", .mixed)
        }
        if state.contains("idle") {
            return ("I", .available)
        }
        return ("?", .neutral)
    }
}

private func compactRuntimeTag(_ raw: String?) -> String? {
    guard let raw else { return nil }
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }

    let daySplit = trimmed.split(separator: "-", maxSplits: 1).map(String.init)
    let days: Int
    let timePart: String

    if daySplit.count == 2 {
        days = Int(daySplit[0]) ?? 0
        timePart = daySplit[1]
    } else {
        days = 0
        timePart = trimmed
    }

    if days > 0 {
        return "\(days)D"
    }

    let components = timePart.split(separator: ":").map(String.init)
    if components.count == 3 {
        let hours = Int(components[0]) ?? 0
        let minutes = Int(components[1]) ?? 0
        if hours > 0 {
            return "\(hours)H"
        }
        return "\(max(1, minutes))M"
    }

    if components.count == 2 {
        let minutes = Int(components[0]) ?? 0
        return "\(max(1, minutes))M"
    }

    return nil
}

private func parseWidgetTimestamp(_ raw: String?) -> Date? {
    guard let raw, !raw.isEmpty else { return nil }

    let fractional = ISO8601DateFormatter()
    fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    if let date = fractional.date(from: raw) {
        return date
    }

    let plain = ISO8601DateFormatter()
    plain.formatOptions = [.withInternetDateTime]
    return plain.date(from: raw)
}

struct SlurmHUDWidgetView: View {
    private struct LayoutSpec {
        let partitionLimit: Int
        let nodeLimit: Int
        let jobLimit: Int
        let partitionColumns: Int
        let nodeColumns: Int
        let microColumns: Int
        let padding: CGFloat
        let sectionSpacing: CGFloat
        let partitionCardPadding: CGFloat
        let nodeCardPadding: CGFloat
        let jobCardPadding: CGFloat
        let headerSize: CGFloat
        let metaSize: CGFloat
        let nodeTitleSize: CGFloat
        let nodeMetaSize: CGFloat
        let jobLineSize: CGFloat
        let textlessOverview: Bool
        let showHeader: Bool
    }

    let entry: SlurmEntry

    @Environment(\.widgetFamily) private var family
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.widgetRenderingMode) private var renderingMode
    @Environment(\.showsWidgetContainerBackground) private var showsBackground

    private var partitionFilter: String? {
        normalizePartitionFilter(entry.widgetPartition)
    }

    private var board: ClusterWidgetBoard? {
        guard let snapshot = entry.snapshot, !snapshot.partitions.isEmpty else { return nil }
        let board = ClusterWidgetBoard(
            snapshot: snapshot,
            lastUpdated: entry.lastUpdated,
            partitionFilter: partitionFilter
        )
        return board.partitions.isEmpty ? nil : board
    }

    private var partitionNotFoundMessage: String? {
        guard let snapshot = entry.snapshot, !snapshot.partitions.isEmpty else { return nil }
        guard let filter = partitionFilter, let rawPartition = entry.widgetPartition else { return nil }

        let hasMatch = snapshot.partitions.contains { partition in
            partitionMatchesFilter(partition.partition ?? "", filter: filter)
        }

        guard !hasMatch else { return nil }
        let trimmed = rawPartition.trimmingCharacters(in: .whitespacesAndNewlines)
        return "Partition \(trimmed) was not found in the cached cluster data."
    }

    private var layout: LayoutSpec {
        switch family {
        case .systemSmall:
            return LayoutSpec(
                partitionLimit: Int.max,
                nodeLimit: Int.max,
                jobLimit: 0,
                partitionColumns: 0,
                nodeColumns: 0,
                microColumns: 6,
                padding: 5,
                sectionSpacing: 3,
                partitionCardPadding: 0,
                nodeCardPadding: 0,
                jobCardPadding: 0,
                headerSize: 0,
                metaSize: 0,
                nodeTitleSize: 0,
                nodeMetaSize: 0,
                jobLineSize: 0,
                textlessOverview: true,
                showHeader: false
            )
        case .systemMedium:
            return LayoutSpec(
                partitionLimit: 4,
                nodeLimit: 4,
                jobLimit: 0,
                partitionColumns: 2,
                nodeColumns: 2,
                microColumns: 0,
                padding: 5,
                sectionSpacing: 4,
                partitionCardPadding: 4,
                nodeCardPadding: 3,
                jobCardPadding: 0,
                headerSize: 7.8,
                metaSize: 6.2,
                nodeTitleSize: 7.2,
                nodeMetaSize: 6.0,
                jobLineSize: 6.8,
                textlessOverview: false,
                showHeader: true
            )
        case .systemLarge:
            return LayoutSpec(
                partitionLimit: Int.max,
                nodeLimit: Int.max,
                jobLimit: 4,
                partitionColumns: 2,
                nodeColumns: 3,
                microColumns: 0,
                padding: 4,
                sectionSpacing: 2.5,
                partitionCardPadding: 3,
                nodeCardPadding: 2,
                jobCardPadding: 2,
                headerSize: 7.8,
                metaSize: 5.9,
                nodeTitleSize: 6.9,
                nodeMetaSize: 5.6,
                jobLineSize: 6.2,
                textlessOverview: false,
                showHeader: true
            )
        case .systemExtraLarge:
            return LayoutSpec(
                partitionLimit: Int.max,
                nodeLimit: Int.max,
                jobLimit: 2,
                partitionColumns: 2,
                nodeColumns: 3,
                microColumns: 0,
                padding: 6,
                sectionSpacing: 4,
                partitionCardPadding: 5,
                nodeCardPadding: 4,
                jobCardPadding: 3,
                headerSize: 8.8,
                metaSize: 6.9,
                nodeTitleSize: 8.2,
                nodeMetaSize: 6.8,
                jobLineSize: 7.5,
                textlessOverview: false,
                showHeader: true
            )
        default:
            return LayoutSpec(
                partitionLimit: Int.max,
                nodeLimit: Int.max,
                jobLimit: 2,
                partitionColumns: 1,
                nodeColumns: 4,
                microColumns: 0,
                padding: 6,
                sectionSpacing: 4,
                partitionCardPadding: 5,
                nodeCardPadding: 4,
                jobCardPadding: 3,
                headerSize: 8.8,
                metaSize: 6.9,
                nodeTitleSize: 8.2,
                nodeMetaSize: 6.8,
                jobLineSize: 7.5,
                textlessOverview: false,
                showHeader: true
            )
        }
    }

    private var compactLargeLayout: Bool {
        family == .systemLarge
    }

    private var compactJobGridSpacing: CGFloat {
        compactLargeLayout ? 1.5 : 2
    }

    private var partitionCardSpacing: CGFloat {
        compactLargeLayout ? 3 : 5
    }

    private var partitionHeaderSpacing: CGFloat {
        compactLargeLayout ? 3 : 4
    }

    private var nodeGridSpacing: CGFloat {
        compactLargeLayout ? 2.5 : 4
    }

    private var nodeCardSpacing: CGFloat {
        compactLargeLayout ? 1.5 : 3
    }

    private var nodeBadgeHorizontalPadding: CGFloat {
        compactLargeLayout ? 3 : 4
    }

    private var nodeBadgeVerticalPadding: CGFloat {
        compactLargeLayout ? 0.5 : 1
    }

    private var partitionCornerRadius: CGFloat {
        compactLargeLayout ? 8 : 10
    }

    private var nodeCornerRadius: CGFloat {
        compactLargeLayout ? 7 : 8
    }

    private var jobCornerRadius: CGFloat {
        compactLargeLayout ? 4 : 5
    }

    private var jobVerticalPadding: CGFloat {
        compactLargeLayout ? 0.35 : 1.1
    }

    private var compactJobVerticalPadding: CGFloat {
        compactLargeLayout ? 0.55 : 0.9
    }

    var body: some View {
        Group {
            if let board {
                if layout.textlessOverview {
                    overviewBoardView(board)
                } else {
                    denseBoardView(board)
                }
            } else if let partitionNotFoundMessage {
                stateView(
                    title: "SlurmHUD",
                    message: partitionNotFoundMessage,
                    symbol: "line.3.horizontal.decrease.circle"
                )
            } else if let error = entry.error {
                stateView(
                    title: "SlurmHUD",
                    message: error,
                    symbol: isWaitingMessage(error) ? "clock" : "exclamationmark.triangle"
                )
            } else {
                stateView(
                    title: "SlurmHUD",
                    message: "Waiting for cached cluster data.",
                    symbol: "clock"
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .containerBackground(for: .widget) {
            WidgetBackdrop(renderingMode: renderingMode, colorScheme: colorScheme)
        }
        .foregroundStyle(primaryTextColor)
    }

    private func overviewBoardView(_ board: ClusterWidgetBoard) -> some View {
        let visiblePartitions = Array(board.partitions.prefix(layout.partitionLimit))

        return VStack(alignment: .leading, spacing: layout.sectionSpacing) {
            ForEach(Array(visiblePartitions.enumerated()), id: \.element.id) { index, partition in
                overviewPartitionRow(partition)
                if index < visiblePartitions.count - 1 {
                    Rectangle()
                        .fill(cardStrokeColor.opacity(0.45))
                        .frame(height: 1)
                }
            }

            Spacer(minLength: 0)
        }
        .padding(layout.padding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func overviewPartitionRow(_ partition: ClusterWidgetBoard.PartitionSection) -> some View {
        let visibleNodes = Array(partition.nodes.prefix(layout.nodeLimit))
        let columns = Array(repeating: GridItem(.flexible(), spacing: 2), count: layout.microColumns)

        return LazyVGrid(columns: columns, spacing: 2) {
            ForEach(visibleNodes) { node in
                microNodeTile(node)
            }
        }
    }

    private func microNodeTile(_ node: ClusterWidgetBoard.NodeTile) -> some View {
        let visibleJobs = min(node.jobs.count, 4)

        return ZStack(alignment: .bottomTrailing) {
            RoundedRectangle(cornerRadius: 3, style: .continuous)
                .fill(nodeFillColor(node.tone))
                .overlay {
                    RoundedRectangle(cornerRadius: 3, style: .continuous)
                        .strokeBorder(nodeStrokeColor(node.tone), lineWidth: 0.8)
                }
                .frame(height: family == .systemSmall ? 12 : 14)

            if visibleJobs > 0 {
                HStack(spacing: 1) {
                    ForEach(0..<visibleJobs, id: \.self) { _ in
                        RoundedRectangle(cornerRadius: 0.8, style: .continuous)
                            .fill(Color.white.opacity(0.95))
                            .frame(width: 2, height: 2)
                    }
                }
                .padding(2)
            }
        }
    }

    private func denseBoardView(_ board: ClusterWidgetBoard) -> some View {
        let visiblePartitions = Array(board.partitions.prefix(layout.partitionLimit))
        let hiddenPartitions = max(0, board.partitions.count - visiblePartitions.count)
        let expandedSinglePartition = visiblePartitions.count == 1
        let columns = Array(
            repeating: GridItem(.flexible(), spacing: layout.sectionSpacing, alignment: .top),
            count: max(1, min(layout.partitionColumns, expandedSinglePartition ? 1 : visiblePartitions.count))
        )
        let summary = "\(visiblePartitions.count)p \(board.totalNodes)n \(board.totalJobs)j"

        return VStack(alignment: .leading, spacing: layout.sectionSpacing) {
            if layout.showHeader {
                widgetHeader(updatedAt: board.refreshedAt, subtitle: summary)
            }

            LazyVGrid(columns: columns, alignment: .leading, spacing: layout.sectionSpacing) {
                ForEach(visiblePartitions) { partition in
                    densePartitionCard(partition, expanded: expandedSinglePartition)
                }
            }

            if hiddenPartitions > 0 {
                overflowText("+\(hiddenPartitions)p")
            }

            Spacer(minLength: 0)
        }
        .padding(layout.padding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func densePartitionCard(_ partition: ClusterWidgetBoard.PartitionSection, expanded: Bool) -> some View {
        let visibleNodes = Array(partition.nodes.prefix(layout.nodeLimit))
        let hiddenNodes = max(0, partition.nodes.count - visibleNodes.count)
        let nodeColumnCount = expanded ? expandedNodeColumnCount : layout.nodeColumns
        let effectiveNodeColumnCount = max(1, min(nodeColumnCount, visibleNodes.count))
        let nodeRowCount = max(1, Int(ceil(Double(max(1, visibleNodes.count)) / Double(effectiveNodeColumnCount))))
        let showAllJobs = hiddenNodes == 0 && nodeRowCount <= verticalJobExpansionRowLimit
        let compactJobColumnCount =
            effectiveNodeColumnCount == 1 ? 4
            : effectiveNodeColumnCount == 2 ? 3
            : 2
        let columns = Array(
            repeating: GridItem(.flexible(), spacing: nodeGridSpacing, alignment: .top),
            count: effectiveNodeColumnCount
        )

        return VStack(alignment: .leading, spacing: partitionCardSpacing) {
            HStack(alignment: .firstTextBaseline, spacing: partitionHeaderSpacing) {
                Text(displayPartitionTag(partition.name))
                    .font(readableWidgetFont(size: layout.headerSize, weight: .bold))
                    .minimumScaleFactor(0.8)
                    .lineLimit(1)
                Spacer(minLength: partitionHeaderSpacing)
                Text(partition.statusText)
                    .font(readableWidgetFont(size: layout.metaSize, weight: .semibold))
                    .foregroundStyle(secondaryTextColor)
                    .minimumScaleFactor(0.8)
                    .lineLimit(1)
            }

            LazyVGrid(columns: columns, spacing: nodeGridSpacing) {
                ForEach(visibleNodes) { node in
                    denseNodeTile(
                        node,
                        showAllJobs: showAllJobs,
                        compactJobColumnCount: compactJobColumnCount
                    )
                }
            }

            if hiddenNodes > 0 {
                overflowText("+\(hiddenNodes)n")
            }
        }
        .padding(layout.partitionCardPadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background {
            RoundedRectangle(cornerRadius: partitionCornerRadius, style: .continuous)
                .fill(cardFillColor)
        }
        .overlay {
            RoundedRectangle(cornerRadius: partitionCornerRadius, style: .continuous)
                .strokeBorder(cardStrokeColor, lineWidth: 0.9)
        }
    }

    @ViewBuilder
    private func denseNodeTile(
        _ node: ClusterWidgetBoard.NodeTile,
        showAllJobs: Bool,
        compactJobColumnCount: Int
    ) -> some View {
        if compactLargeLayout {
            compactLargeNodeTile(
                node,
                showAllJobs: showAllJobs,
                compactJobColumnCount: compactJobColumnCount
            )
        } else {
            regularDenseNodeTile(node, showAllJobs: showAllJobs)
        }
    }

    private func regularDenseNodeTile(_ node: ClusterWidgetBoard.NodeTile, showAllJobs: Bool) -> some View {
        let visibleJobs = showAllJobs ? node.jobs : Array(node.jobs.prefix(layout.jobLimit))
        let hiddenJobs = max(0, node.jobs.count - visibleJobs.count)

        return VStack(alignment: .leading, spacing: nodeCardSpacing) {
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(shortNodeTag(node.name))
                    .font(readableWidgetFont(size: layout.nodeTitleSize, weight: .semibold))
                    .minimumScaleFactor(0.8)
                    .lineLimit(1)
                Spacer(minLength: 2)
                Text(node.stateLabel)
                    .font(readableWidgetFont(size: layout.metaSize, weight: .bold))
                    .foregroundStyle(primaryTextColor)
                    .padding(.horizontal, nodeBadgeHorizontalPadding)
                    .padding(.vertical, nodeBadgeVerticalPadding)
                    .background {
                        Capsule(style: .continuous)
                            .fill(nodeStateBadgeFill(node.tone))
                    }
            }

            if !node.gpuLabel.isEmpty, node.gpuLabel != "CPU" {
                Text(shortGPUTag(node.gpuLabel))
                    .font(readableWidgetFont(size: layout.nodeMetaSize, weight: .medium))
                    .foregroundStyle(secondaryTextColor)
                    .minimumScaleFactor(0.75)
                    .lineLimit(1)
            }

            ForEach(visibleJobs) { job in
                denseJobCard(job)
            }

            if hiddenJobs > 0 {
                Text("+\(hiddenJobs)")
                    .font(readableWidgetFont(size: layout.jobLineSize, weight: .medium))
                    .foregroundStyle(secondaryTextColor)
                    .minimumScaleFactor(0.8)
                    .lineLimit(1)
            }
        }
        .padding(layout.nodeCardPadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background {
            RoundedRectangle(cornerRadius: nodeCornerRadius, style: .continuous)
                .fill(nodeFillColor(node.tone))
        }
        .overlay {
            RoundedRectangle(cornerRadius: nodeCornerRadius, style: .continuous)
                .strokeBorder(nodeStrokeColor(node.tone), lineWidth: 0.8)
        }
    }

    private func compactLargeNodeTile(
        _ node: ClusterWidgetBoard.NodeTile,
        showAllJobs: Bool,
        compactJobColumnCount: Int
    ) -> some View {
        let visibleJobs = showAllJobs ? node.jobs : Array(node.jobs.prefix(layout.jobLimit))
        let hiddenJobs = max(0, node.jobs.count - visibleJobs.count)
        let jobCount = visibleJobs.count + (hiddenJobs > 0 ? 1 : 0)
        let jobColumns = Array(
            repeating: GridItem(.flexible(), spacing: compactJobGridSpacing, alignment: .top),
            count: max(1, min(compactJobColumnCount, jobCount))
        )

        return VStack(alignment: .leading, spacing: nodeCardSpacing) {
            Text(compactNodeLine(node))
                .font(readableWidgetFont(size: layout.nodeTitleSize, weight: .semibold))
                .foregroundStyle(primaryTextColor)
                .minimumScaleFactor(0.75)
                .lineLimit(1)

            if !visibleJobs.isEmpty || hiddenJobs > 0 {
                LazyVGrid(columns: jobColumns, spacing: compactJobGridSpacing) {
                    ForEach(visibleJobs) { job in
                        compactJobCube(job)
                    }
                    if hiddenJobs > 0 {
                        compactOverflowCube("+\(hiddenJobs)")
                    }
                }
            }
        }
        .padding(layout.nodeCardPadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background {
            RoundedRectangle(cornerRadius: nodeCornerRadius, style: .continuous)
                .fill(nodeFillColor(node.tone))
        }
        .overlay {
            RoundedRectangle(cornerRadius: nodeCornerRadius, style: .continuous)
                .strokeBorder(nodeStrokeColor(node.tone), lineWidth: 0.9)
        }
    }

    private func denseJobCard(_ job: ClusterWidgetBoard.JobTile) -> some View {
        Text(denseJobLine(job))
            .font(readableWidgetFont(size: layout.jobLineSize, weight: .medium))
            .foregroundStyle(secondaryTextColor)
            .minimumScaleFactor(0.75)
            .lineLimit(1)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, layout.jobCardPadding)
            .padding(.vertical, jobVerticalPadding)
            .background {
                RoundedRectangle(cornerRadius: jobCornerRadius, style: .continuous)
                    .fill(jobFillColor)
            }
            .overlay {
                RoundedRectangle(cornerRadius: jobCornerRadius, style: .continuous)
                    .strokeBorder(jobStrokeColor, lineWidth: 0.7)
            }
    }

    private func compactJobCube(_ job: ClusterWidgetBoard.JobTile) -> some View {
        Text(compactJobLine(job))
            .font(readableWidgetFont(size: layout.jobLineSize, weight: .semibold))
            .foregroundStyle(primaryTextColor)
            .minimumScaleFactor(0.52)
            .lineLimit(1)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .padding(.horizontal, 2)
        .padding(.vertical, compactJobVerticalPadding)
        .background {
            RoundedRectangle(cornerRadius: jobCornerRadius, style: .continuous)
                .fill(jobFillColor)
        }
        .overlay {
            RoundedRectangle(cornerRadius: jobCornerRadius, style: .continuous)
                .strokeBorder(jobStrokeColor, lineWidth: 0.7)
        }
    }

    private func compactOverflowCube(_ label: String) -> some View {
        Text(label)
            .font(readableWidgetFont(size: layout.metaSize, weight: .semibold))
            .foregroundStyle(secondaryTextColor)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.horizontal, 2)
            .padding(.vertical, compactJobVerticalPadding)
            .background {
                RoundedRectangle(cornerRadius: jobCornerRadius, style: .continuous)
                    .fill(jobFillColor)
            }
            .overlay {
                RoundedRectangle(cornerRadius: jobCornerRadius, style: .continuous)
                    .strokeBorder(jobStrokeColor, lineWidth: 0.7)
            }
    }

    private func widgetHeader(updatedAt: Date?, subtitle: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Text(subtitle)
                .font(readableWidgetFont(size: layout.headerSize, weight: .semibold))
                .minimumScaleFactor(0.8)
                .lineLimit(1)
            Spacer(minLength: 2)
            if let updatedAt {
                Text(updatedAt, style: .relative)
                    .font(readableWidgetFont(size: layout.metaSize, weight: .medium))
                    .foregroundStyle(secondaryTextColor)
                    .minimumScaleFactor(0.8)
            }
        }
    }

    private func stateView(title: String, message: String, symbol: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: symbol)
                .font(readableWidgetFont(size: 13, weight: .semibold))
            Text(message)
                .font(readableWidgetFont(size: 11, weight: .regular))
                .foregroundStyle(secondaryTextColor)
                .lineLimit(family == .systemSmall ? 4 : 6)
            Spacer(minLength: 0)
            Text(stateFootnote(for: message))
                .font(readableWidgetFont(size: 10, weight: .regular))
                .foregroundStyle(secondaryTextColor)
        }
        .padding(layout.padding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func overflowText(_ message: String) -> some View {
        Text(message)
            .font(readableWidgetFont(size: layout.metaSize, weight: .medium))
            .foregroundStyle(secondaryTextColor)
            .lineLimit(1)
    }

    private var expandedNodeColumnCount: Int {
        switch family {
        case .systemLarge:
            return 3
        case .systemExtraLarge:
            return 4
        default:
            return layout.nodeColumns
        }
    }

    private var verticalJobExpansionRowLimit: Int {
        switch family {
        case .systemLarge:
            return 2
        case .systemExtraLarge:
            return 3
        default:
            return 2
        }
    }

    private func readableWidgetFont(size: CGFloat, weight: Font.Weight) -> Font {
        .custom("SF Compact Text", size: size).weight(weight)
    }

    private func denseJobLine(_ job: ClusterWidgetBoard.JobTile) -> String {
        let resource = job.resourceLine == "-" ? shortJobTag(job.name) : job.resourceLine
        return [job.userTag.uppercased(), resource, job.runtimeTag]
            .compactMap { $0 }
            .joined(separator: " ")
    }

    private func compactJobLine(_ job: ClusterWidgetBoard.JobTile) -> String {
        let resource = job.compactResourceLine.isEmpty ? shortJobTag(job.name) : job.compactResourceLine
        return [job.userTag.uppercased(), resource, job.runtimeTag]
            .compactMap { $0 }
            .joined(separator: " ")
    }

    private func compactTag(_ raw: String, limit: Int) -> String {
        let cleaned = raw
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .filter { !$0.isEmpty }

        if cleaned.count > 1 {
            let joined = cleaned
                .map { String($0.prefix(1)).uppercased() }
                .joined()
            if joined.count >= min(2, limit) {
                return String(joined.prefix(limit))
            }
        }

        let compact = raw
            .filter { $0.isLetter || $0.isNumber }
        return String(compact.prefix(limit)).uppercased()
    }

    private func shortPartitionTag(_ raw: String) -> String {
        compactTag(raw, limit: 4)
    }

    private func displayPartitionTag(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return shortPartitionTag(raw) }

        let compactUnderscored = trimmed
            .lowercased()
            .replacingOccurrences(of: " ", with: "_")

        if compactUnderscored.count <= 8 {
            return compactUnderscored
        }

        if compactLargeLayout {
            return String(compactUnderscored.prefix(8))
        }

        return shortPartitionTag(raw)
    }

    private func shortNodeTag(_ raw: String) -> String {
        let digits = String(raw.reversed().prefix { $0.isNumber }.reversed())
        if let first = raw.first, !digits.isEmpty {
            return "\(String(first).lowercased())\(digits)"
        }
        return compactTag(raw, limit: 4).lowercased()
    }

    private func shortGPUTag(_ raw: String) -> String {
        compactTag(raw, limit: 5)
    }

    private func shortJobTag(_ raw: String) -> String {
        compactTag(raw, limit: 5).lowercased()
    }

    private func compactNodeLine(_ node: ClusterWidgetBoard.NodeTile) -> String {
        let nodeTag = shortNodeTag(node.name)
        let rawGPU = node.gpuLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        let gpuTag = shortGPUTag(rawGPU)
        let hideGPU =
            rawGPU.isEmpty
            || rawGPU.caseInsensitiveCompare("CPU") == .orderedSame
            || rawGPU.localizedCaseInsensitiveContains("await")

        guard !hideGPU, !gpuTag.isEmpty else {
            return nodeTag
        }
        return "\(nodeTag) \(gpuTag)"
    }

    private func isWaitingMessage(_ message: String) -> Bool {
        message.localizedCaseInsensitiveContains("next scheduled refresh")
    }

    private func stateFootnote(for message: String) -> String {
        if isWaitingMessage(message) {
            return "The widget will update on the next refresh cycle."
        }
        return "Launch the app to refresh the shared cache."
    }

    private var usesCustomBackdrop: Bool {
        showsBackground && renderingMode != .vibrant
    }

    private var primaryTextColor: Color {
        if usesCustomBackdrop {
            return colorScheme == .dark
                ? Color.white.opacity(0.96)
                : Color(red: 0.12, green: 0.16, blue: 0.23)
        }
        return .primary
    }

    private var secondaryTextColor: Color {
        if usesCustomBackdrop {
            return colorScheme == .dark
                ? Color.white.opacity(0.74)
                : Color(red: 0.23, green: 0.30, blue: 0.39).opacity(0.88)
        }
        return .secondary
    }

    private var cardFillColor: Color {
        guard showsBackground else { return .clear }
        if renderingMode == .vibrant {
            return colorScheme == .dark ? Color.white.opacity(0.12) : Color.white.opacity(0.28)
        }
        return colorScheme == .dark
            ? Color.white.opacity(0.15)
            : Color.white.opacity(0.52)
    }

    private var cardStrokeColor: Color {
        if colorScheme == .dark {
            return Color.white.opacity(renderingMode == .vibrant ? 0.22 : 0.16)
        }
        return Color.white.opacity(renderingMode == .vibrant ? 0.24 : 0.42)
    }

    private var jobFillColor: Color {
        if usesCustomBackdrop {
            return colorScheme == .dark
                ? Color.white.opacity(0.09)
                : Color.white.opacity(0.44)
        }
        return Color.primary.opacity(0.05)
    }

    private var jobStrokeColor: Color {
        if usesCustomBackdrop {
            return colorScheme == .dark
                ? Color.white.opacity(0.12)
                : Color.white.opacity(0.32)
        }
        return Color.primary.opacity(0.08)
    }

    private func nodeFillColor(_ tone: ClusterWidgetBoard.NodeTile.Tone) -> Color {
        let strongBackdrop = usesCustomBackdrop && colorScheme == .dark

        switch tone {
        case .available:
            return Color.green.opacity(strongBackdrop ? 0.18 : 0.12)
        case .mixed:
            return Color.cyan.opacity(strongBackdrop ? 0.16 : 0.12)
        case .busy:
            return Color.orange.opacity(strongBackdrop ? 0.18 : 0.12)
        case .warning:
            return Color.red.opacity(strongBackdrop ? 0.18 : 0.12)
        case .neutral:
            if usesCustomBackdrop {
                return colorScheme == .dark ? Color.white.opacity(0.11) : Color.white.opacity(0.36)
            }
            return Color.primary.opacity(0.04)
        }
    }

    private func nodeStrokeColor(_ tone: ClusterWidgetBoard.NodeTile.Tone) -> Color {
        switch tone {
        case .available:
            return Color.green.opacity(usesCustomBackdrop && colorScheme == .dark ? 0.50 : 0.26)
        case .mixed:
            return Color.cyan.opacity(usesCustomBackdrop && colorScheme == .dark ? 0.48 : 0.26)
        case .busy:
            return Color.orange.opacity(usesCustomBackdrop && colorScheme == .dark ? 0.50 : 0.26)
        case .warning:
            return Color.red.opacity(usesCustomBackdrop && colorScheme == .dark ? 0.50 : 0.26)
        case .neutral:
            if usesCustomBackdrop {
                return colorScheme == .dark ? Color.white.opacity(0.12) : Color.white.opacity(0.28)
            }
            return Color.primary.opacity(0.10)
        }
    }

    private func nodeStateBadgeFill(_ tone: ClusterWidgetBoard.NodeTile.Tone) -> Color {
        switch tone {
        case .available:
            return Color.green.opacity(0.24)
        case .mixed:
            return Color.cyan.opacity(0.22)
        case .busy:
            return Color.orange.opacity(0.24)
        case .warning:
            return Color.red.opacity(0.24)
        case .neutral:
            return usesCustomBackdrop ? Color.white.opacity(0.14) : Color.primary.opacity(0.08)
        }
    }
}

private struct WidgetBackdrop: View {
    let renderingMode: WidgetRenderingMode
    let colorScheme: ColorScheme

    var body: some View {
        if renderingMode == .vibrant {
            Color.clear
        } else {
            ZStack {
                if colorScheme == .dark {
                    Rectangle()
                        .fill(.thinMaterial)
                } else {
                    Rectangle()
                        .fill(.regularMaterial)
                }

                Rectangle()
                    .fill(
                        LinearGradient(
                            colors: colorScheme == .dark
                                ? [
                                    Color.white.opacity(0.035),
                                    Color.white.opacity(0.008)
                                ]
                                : [
                                    Color.white.opacity(0.18),
                                    Color.white.opacity(0.04)
                                ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )

                Rectangle()
                    .strokeBorder(
                        colorScheme == .dark
                            ? Color.white.opacity(0.16)
                            : Color.white.opacity(0.42),
                        lineWidth: 1
                    )
            }
        }
    }
}

@main
struct SlurmHUDWidget: Widget {
    private let kind = "SlurmHUDWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: SlurmProvider()) { entry in
            SlurmHUDWidgetView(entry: entry)
        }
        .configurationDisplayName("SlurmHUD")
        .description("Cluster availability at a glance.")
        .supportedFamilies([.systemSmall, .systemMedium, .systemLarge, .systemExtraLarge])
    }
}
