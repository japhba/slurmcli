import Foundation

struct SlurmHUDConfig: Codable {
    var host: String
    var refreshSeconds: Int
    var timeoutSeconds: Int
    var command: String

    static let `default` = SlurmHUDConfig(
        host: "mycluster",
        refreshSeconds: 600,
        timeoutSeconds: 10,
        command: "slurmcli-status -vv"
    )
}

// File paths for config and cache (accessible to both app and widget on macOS)
private let configFileName = ".slurmhud.json"
private let cacheFileName = ".slurmhud-cache.json"

private func homeDirectory() -> URL {
    // Use HOME environment variable for reliable access from widget
    if let home = ProcessInfo.processInfo.environment["HOME"] {
        return URL(fileURLWithPath: home)
    }
    return FileManager.default.homeDirectoryForCurrentUser
}

final class ConfigStore: ObservableObject {
    @Published var config: SlurmHUDConfig

    init() {
        self.config = ConfigStore.loadFromFile() ?? .default
    }

    func save() {
        let url = homeDirectory().appendingPathComponent(configFileName)
        if let data = try? JSONEncoder().encode(config) {
            try? data.write(to: url)
        }
    }

    func reload() {
        if let loaded = ConfigStore.loadFromFile() {
            config = loaded
        }
    }

    private static func loadFromFile() -> SlurmHUDConfig? {
        let url = homeDirectory().appendingPathComponent(configFileName)
        guard let data = try? Data(contentsOf: url) else { return nil }
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
        homeDirectory().appendingPathComponent(cacheFileName)
    }

    static func save(snapshot: ClusterSnapshot, error: String? = nil) {
        let cached = CachedSnapshot(snapshot: snapshot, fetchedAt: Date(), error: error)
        if let data = try? JSONEncoder().encode(cached) {
            try? data.write(to: cacheURL)
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
            try? data.write(to: cacheURL)
        }
    }

    static func load() -> CachedSnapshot? {
        guard let data = try? Data(contentsOf: cacheURL) else { return nil }
        guard let cached = try? JSONDecoder().decode(CachedSnapshot.self, from: data) else { return nil }
        // Only return if we have actual data (not just an error)
        if cached.snapshot.partitions.isEmpty && cached.error != nil {
            return nil
        }
        return cached
    }

    static func loadError() -> (error: String, fetchedAt: Date)? {
        guard let data = try? Data(contentsOf: cacheURL) else { return nil }
        guard let cached = try? JSONDecoder().decode(CachedSnapshot.self, from: data) else { return nil }
        // Return error info if snapshot is empty (error-only cache)
        if cached.snapshot.partitions.isEmpty, let error = cached.error {
            return (error, cached.fetchedAt)
        }
        return nil
    }
}
