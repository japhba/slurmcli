import Foundation

struct SlurmHUDConfig: Codable {
    var host: String
    var refreshSeconds: Int
    var timeoutSeconds: Int
    var command: String
    var widgetPartition: String?
    var useMockData: Bool
    var backgroundRefreshEnabled: Bool

    init(
        host: String,
        refreshSeconds: Int,
        timeoutSeconds: Int,
        command: String,
        widgetPartition: String? = nil,
        useMockData: Bool = false,
        backgroundRefreshEnabled: Bool = true
    ) {
        self.host = host
        self.refreshSeconds = refreshSeconds
        self.timeoutSeconds = timeoutSeconds
        self.command = command
        self.widgetPartition = Self.normalizedWidgetPartition(widgetPartition)
        self.useMockData = useMockData
        self.backgroundRefreshEnabled = backgroundRefreshEnabled
    }

    static let `default` = SlurmHUDConfig(
        host: "mycluster",
        refreshSeconds: 600,
        timeoutSeconds: 180,
        command: "slurmcli-status -vvvv",
        widgetPartition: "gpu_lowp",
        useMockData: true,
        backgroundRefreshEnabled: true
    )

    private enum CodingKeys: String, CodingKey {
        case host
        case refreshSeconds
        case timeoutSeconds
        case command
        case widgetPartition
        case useMockData
        case backgroundRefreshEnabled
    }

    private enum LegacyCodingKeys: String, CodingKey {
        case host
        case refresh_seconds
        case timeout_seconds
        case command
        case widget_partition
        case use_mock_data
        case background_refresh_enabled
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
        self.backgroundRefreshEnabled =
            try container.decodeIfPresent(Bool.self, forKey: .backgroundRefreshEnabled)
            ?? legacyContainer.decodeIfPresent(Bool.self, forKey: .background_refresh_enabled)
            ?? Self.default.backgroundRefreshEnabled
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(host, forKey: .host)
        try container.encode(refreshSeconds, forKey: .refreshSeconds)
        try container.encode(timeoutSeconds, forKey: .timeoutSeconds)
        try container.encode(command, forKey: .command)
        try container.encode(widgetPartition ?? "", forKey: .widgetPartition)
        try container.encode(useMockData, forKey: .useMockData)
        try container.encode(backgroundRefreshEnabled, forKey: .backgroundRefreshEnabled)
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

/// Single source of truth for the App Group container path.
///
/// We resolve exactly one identifier (from `Bundle.main`'s Info.plist,
/// falling back to a hardcoded default only if that's missing) and use
/// `FileManager.containerURL(forSecurityApplicationGroupIdentifier:)` to
/// get the canonical container URL for the running process. No fallback
/// to a manually-constructed `~/Library/Group Containers/X` path, no
/// iteration over multiple bundles' identifiers, no legacy `~/.slurmhud*`
/// paths, no multi-path writes.
///
/// All those fallbacks were what caused the recurring "SlurmHUD would
/// like to access data from other apps" TCC prompts: every time a
/// non-entitled path was touched (because the running process's
/// entitlement only covers its own group ID), macOS asked for permission.
/// One identifier, one entitlement, one container — no prompts.
private enum SharedContainer {
    private static let fileManager = FileManager.default

    /// One-time migration from any historical container layouts to the
    /// resolved one. Done lazily on first access so the (potentially TCC-
    /// prompting) read of legacy paths only happens once per install.
    private static let migrate: Void = {
        runMigration()
    }()

    static var groupIdentifier: String {
        if let raw = Bundle.main.object(forInfoDictionaryKey: appGroupInfoKey) as? String {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty, !trimmed.contains("$(") {
                return trimmed
            }
        }
        return defaultAppGroupIdentifier
    }

    /// Container URL without triggering migration. Used by both the
    /// public accessors and `runMigration` itself, so migration can
    /// resolve the destination path without re-entering the lazy
    /// initializer that drives it.
    private static var rawContainerURL: URL {
        if let url = fileManager.containerURL(forSecurityApplicationGroupIdentifier: groupIdentifier) {
            return url
        }
        // Last-resort path. If we ever land here it means the current
        // process has no App Group entitlement — accessing this path will
        // most likely fail. We deliberately don't try alternate IDs;
        // failing visibly is better than dragging in a TCC prompt.
        return fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Group Containers", isDirectory: true)
            .appendingPathComponent(groupIdentifier, isDirectory: true)
    }

    static var containerURL: URL {
        _ = migrate
        return rawContainerURL
    }

    static func configURL() -> URL {
        containerURL.appendingPathComponent(configFileName, isDirectory: false)
    }

    static func cacheURL() -> URL {
        containerURL.appendingPathComponent(cacheFileName, isDirectory: false)
    }

    static func loadConfigData() -> Data? {
        try? Data(contentsOf: configURL())
    }

    static func saveConfigData(_ data: Data) {
        writeAtomically(data, to: configURL())
    }

    static func loadCacheData() -> Data? {
        try? Data(contentsOf: cacheURL())
    }

    static func saveCacheData(_ data: Data) {
        writeAtomically(data, to: cacheURL())
    }

    private static func writeAtomically(_ data: Data, to url: URL) {
        do {
            try fileManager.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: url, options: .atomic)
        } catch {
            NSLog("SharedContainer: failed to write \(url.path): \(error)")
        }
    }

    /// One-time copy of config from any pre-existing container layout.
    /// We deliberately only migrate the config (small, painful to lose),
    /// not the cache (ephemeral, gets rewritten on the next refresh).
    private static func runMigration() {
        let target = rawContainerURL.appendingPathComponent(configFileName, isDirectory: false)
        guard !fileManager.fileExists(atPath: target.path) else { return }

        let home = fileManager.homeDirectoryForCurrentUser
        let groupContainers = home.appendingPathComponent("Library/Group Containers", isDirectory: true)
        var sources: [URL] = []

        // Historical sibling Group Containers (e.g. team-prefixed
        // 8XAD7C8Z68.com.jbauer.slurmhud.shared) that an earlier version
        // of the app created on this machine.
        if let entries = try? fileManager.contentsOfDirectory(
            at: groupContainers,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) {
            for entry in entries {
                if entry.lastPathComponent.hasSuffix(".com.jbauer.slurmhud.shared")
                    || entry.lastPathComponent == "com.jbauer.slurmhud.shared" {
                    let candidate = entry.appendingPathComponent(configFileName, isDirectory: false)
                    if candidate.standardizedFileURL.path != target.standardizedFileURL.path {
                        sources.append(candidate)
                    }
                }
            }
        }

        // Pre-App-Group fallback that very old builds wrote to.
        sources.append(home.appendingPathComponent(configFileName))

        for source in sources {
            guard fileManager.fileExists(atPath: source.path),
                  let data = try? Data(contentsOf: source) else { continue }
            writeAtomically(data, to: target)
            NSLog("SharedContainer: migrated config from \(source.path) -> \(target.path)")
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
    static var cacheURL: URL {
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
        try? FileManager.default.removeItem(at: SharedContainer.cacheURL())
    }
}
