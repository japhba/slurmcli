import Foundation

struct SlurmHUDConfig: Codable {
    var host: String
    var refreshSeconds: Int
    var timeoutSeconds: Int
    var command: String
    var widgetPartition: String?
    var useMockData: Bool

    init(
        host: String,
        refreshSeconds: Int,
        timeoutSeconds: Int,
        command: String,
        widgetPartition: String? = nil,
        useMockData: Bool = false
    ) {
        self.host = host
        self.refreshSeconds = refreshSeconds
        self.timeoutSeconds = timeoutSeconds
        self.command = command
        self.widgetPartition = Self.normalizedWidgetPartition(widgetPartition)
        self.useMockData = useMockData
    }

    static let `default` = SlurmHUDConfig(
        host: "mycluster",
        refreshSeconds: 600,
        timeoutSeconds: 120,
        command: "slurmcli-status -vvvv",
        widgetPartition: "gpu_lowp",
        useMockData: true
    )

    private enum CodingKeys: String, CodingKey {
        case host
        case refreshSeconds
        case timeoutSeconds
        case command
        case widgetPartition
        case useMockData
    }

    private enum LegacyCodingKeys: String, CodingKey {
        case host
        case refresh_seconds
        case timeout_seconds
        case command
        case widget_partition
        case use_mock_data
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let legacyContainer = try decoder.container(keyedBy: LegacyCodingKeys.self)

        self.host =
            try container.decodeIfPresent(String.self, forKey: .host)
            ?? legacyContainer.decodeIfPresent(String.self, forKey: .host)
            ?? Self.default.host
        self.refreshSeconds =
            try container.decodeIfPresent(Int.self, forKey: .refreshSeconds)
            ?? legacyContainer.decodeIfPresent(Int.self, forKey: .refresh_seconds)
            ?? Self.default.refreshSeconds
        self.timeoutSeconds =
            try container.decodeIfPresent(Int.self, forKey: .timeoutSeconds)
            ?? legacyContainer.decodeIfPresent(Int.self, forKey: .timeout_seconds)
            ?? Self.default.timeoutSeconds
        self.command =
            try container.decodeIfPresent(String.self, forKey: .command)
            ?? legacyContainer.decodeIfPresent(String.self, forKey: .command)
            ?? Self.default.command
        let hasExplicitWidgetPartition =
            container.contains(.widgetPartition) || legacyContainer.contains(.widget_partition)
        let decodedWidgetPartition =
            try container.decodeIfPresent(String.self, forKey: .widgetPartition)
            ?? legacyContainer.decodeIfPresent(String.self, forKey: .widget_partition)
        self.widgetPartition =
            hasExplicitWidgetPartition
            ? Self.normalizedWidgetPartition(decodedWidgetPartition)
            : Self.default.widgetPartition
        self.useMockData =
            try container.decodeIfPresent(Bool.self, forKey: .useMockData)
            ?? legacyContainer.decodeIfPresent(Bool.self, forKey: .use_mock_data)
            ?? false
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(host, forKey: .host)
        try container.encode(refreshSeconds, forKey: .refreshSeconds)
        try container.encode(timeoutSeconds, forKey: .timeoutSeconds)
        try container.encode(command, forKey: .command)
        try container.encode(widgetPartition ?? "", forKey: .widgetPartition)
        try container.encode(useMockData, forKey: .useMockData)
    }

    private static func normalizedWidgetPartition(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

private let appGroupInfoKey = "SlurmHUDAppGroupIdentifier"
private let defaultAppGroupIdentifier = "com.jbauer.slurmhud.shared"
private let configFileName = ".slurmhud.json"
private let cacheFileName = ".slurmhud-cache.json"

private func homeDirectory() -> URL {
    let home = FileManager.default.homeDirectoryForCurrentUser
    if !home.path.isEmpty {
        return home
    }
    if let environmentHome = ProcessInfo.processInfo.environment["HOME"] {
        return URL(fileURLWithPath: environmentHome)
    }
    return URL(fileURLWithPath: NSHomeDirectory())
}

private enum SharedContainer {
    private static let fileManager = FileManager.default

    static func configURL() -> URL {
        preferredURL(for: configFileName)
    }

    static func cacheURL() -> URL {
        preferredURL(for: cacheFileName)
    }

    static func configURLs() -> [URL] {
        candidateURLs(for: configFileName)
    }

    static func cacheURLs() -> [URL] {
        candidateURLs(for: cacheFileName)
    }

    static func loadConfigData() -> Data? {
        loadMostRecentData(from: configURLs())
    }

    static func saveConfigData(_ data: Data) {
        writeData(data, to: configURLs())
    }

    static func loadCacheData() -> Data? {
        loadMostRecentData(from: cacheURLs())
    }

    static func saveCacheData(_ data: Data) {
        writeData(data, to: cacheURLs())
    }

    private static func preferredURL(for fileName: String) -> URL {
        candidateURLs(for: fileName).first ?? legacyURL(for: fileName)
    }

    private static func candidateURLs(for fileName: String) -> [URL] {
        let legacyURL = legacyURL(for: fileName)
        var urls: [URL] = []

        for groupID in appGroupIdentifiers {
            if let containerURL = fileManager.containerURL(forSecurityApplicationGroupIdentifier: groupID) {
                let sharedURL = containerURL.appendingPathComponent(fileName, isDirectory: false)
                migrateIfNeeded(from: legacyURL, to: sharedURL)
                appendUnique(sharedURL, to: &urls)
            }

            let directURL = homeDirectory()
                .appendingPathComponent("Library/Group Containers", isDirectory: true)
                .appendingPathComponent(groupID, isDirectory: true)
                .appendingPathComponent(fileName, isDirectory: false)
            migrateIfNeeded(from: legacyURL, to: directURL)
            appendUnique(directURL, to: &urls)
        }

        appendUnique(legacyURL, to: &urls)
        return urls
    }

    private static var appGroupIdentifiers: [String] {
        let bundles = [Bundle.main] + Bundle.allBundles + Bundle.allFrameworks
        var identifiers: [String] = []

        for bundle in bundles {
            guard let raw = bundle.object(forInfoDictionaryKey: appGroupInfoKey) as? String else {
                continue
            }

            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !trimmed.contains("$(") else {
                continue
            }

            if !identifiers.contains(trimmed) {
                identifiers.append(trimmed)
            }
        }

        if !identifiers.contains(defaultAppGroupIdentifier) {
            identifiers.append(defaultAppGroupIdentifier)
        }

        return identifiers
    }

    private static func legacyURL(for fileName: String) -> URL {
        homeDirectory().appendingPathComponent(fileName)
    }

    private static func appendUnique(_ url: URL, to urls: inout [URL]) {
        let path = url.standardizedFileURL.path
        guard !urls.contains(where: { $0.standardizedFileURL.path == path }) else { return }
        urls.append(url)
    }

    private static func loadMostRecentData(from urls: [URL]) -> Data? {
        let orderedCandidates = urls.enumerated()
            .filter { fileManager.fileExists(atPath: $0.element.path) }
            .sorted { lhs, rhs in
                let lhsDate = modificationDate(for: lhs.element)
                let rhsDate = modificationDate(for: rhs.element)
                if lhsDate != rhsDate {
                    return lhsDate > rhsDate
                }
                return lhs.offset < rhs.offset
            }

        for (_, url) in orderedCandidates {
            if let data = try? Data(contentsOf: url) {
                return data
            }
        }

        return nil
    }

    private static func modificationDate(for url: URL) -> Date {
        let values = try? url.resourceValues(forKeys: [.contentModificationDateKey])
        return values?.contentModificationDate ?? .distantPast
    }

    private static func writeData(_ data: Data, to urls: [URL]) {
        for url in urls {
            do {
                try fileManager.createDirectory(
                    at: url.deletingLastPathComponent(),
                    withIntermediateDirectories: true,
                    attributes: nil
                )
                try data.write(to: url, options: .atomic)
            } catch {
                continue
            }
        }
    }

    private static func migrateIfNeeded(from legacyURL: URL, to sharedURL: URL) {
        guard !fileManager.fileExists(atPath: sharedURL.path) else { return }
        guard fileManager.fileExists(atPath: legacyURL.path) else { return }

        do {
            try fileManager.createDirectory(
                at: sharedURL.deletingLastPathComponent(),
                withIntermediateDirectories: true,
                attributes: nil
            )
            try fileManager.copyItem(at: legacyURL, to: sharedURL)
        } catch {
            return
        }
    }
}

final class ConfigStore: ObservableObject {
    @Published var config: SlurmHUDConfig

    init() {
        self.config = ConfigStore.loadShared() ?? .default
    }

    func save() {
        if let data = try? JSONEncoder().encode(config) {
            SharedContainer.saveConfigData(data)
        }
    }

    func reload() {
        if let loaded = ConfigStore.loadShared() {
            config = loaded
        }
    }

    static func loadShared() -> SlurmHUDConfig? {
        guard let data = SharedContainer.loadConfigData() else { return nil }
        return try? JSONDecoder().decode(SlurmHUDConfig.self, from: data)
    }
}

// Cache for storing snapshot data (file-based for widget access)
struct CachedSnapshot: Codable {
    let snapshot: ClusterSnapshot
    let fetchedAt: Date
    let error: String?
}

final class SnapshotCache {
    private static var cacheURL: URL {
        SharedContainer.cacheURL()
    }

    static func save(snapshot: ClusterSnapshot, error: String? = nil) {
        let cached = CachedSnapshot(snapshot: snapshot, fetchedAt: Date(), error: error)
        if let data = try? JSONEncoder().encode(cached) {
            SharedContainer.saveCacheData(data)
        }
    }

    static func saveError(_ error: String) {
        // For errors, we save a cache with nil snapshot
        let cached = CachedSnapshot(
            snapshot: ClusterSnapshot(generated_at: nil, partitions: []),
            fetchedAt: Date(),
            error: error
        )
        if let data = try? JSONEncoder().encode(cached) {
            SharedContainer.saveCacheData(data)
        }
    }

    static func load() -> CachedSnapshot? {
        guard let data = SharedContainer.loadCacheData() else { return nil }
        guard let cached = try? JSONDecoder().decode(CachedSnapshot.self, from: data) else { return nil }
        // Only return if we have actual data (not just an error)
        if cached.snapshot.partitions.isEmpty && cached.error != nil {
            return nil
        }
        return cached
    }

    static func loadError() -> (error: String, fetchedAt: Date)? {
        guard let data = SharedContainer.loadCacheData() else { return nil }
        guard let cached = try? JSONDecoder().decode(CachedSnapshot.self, from: data) else { return nil }
        // Return error info if snapshot is empty (error-only cache)
        if cached.snapshot.partitions.isEmpty, let error = cached.error {
            return (error, cached.fetchedAt)
        }
        return nil
    }

    static func sharedAccessErrorMessage() -> String? {
        if let message = readAccessIssueMessage(for: cacheURL) {
            return message
        }
        return readAccessIssueMessage(for: SharedContainer.configURL())
    }

    private static func readAccessIssueMessage(for url: URL) -> String? {
        do {
            _ = try Data(contentsOf: url)
            return nil
        } catch {
            let nsError = error as NSError
            let isPermissionError =
                nsError.domain == NSCocoaErrorDomain && nsError.code == NSFileReadNoPermissionError

            guard isPermissionError else { return nil }

            return "Widget cannot access shared data. In Xcode, sign SlurmHUD and SlurmHUDWidget with the same Development Team and App Group."
        }
    }

    static func clear() {
        for url in SharedContainer.cacheURLs() {
            try? FileManager.default.removeItem(at: url)
        }
    }
}
