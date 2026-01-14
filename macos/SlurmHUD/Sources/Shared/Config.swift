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

final class ConfigStore: ObservableObject {
    @Published var config: SlurmHUDConfig

    private let fileURL: URL

    init() {
        self.fileURL = ConfigStore.defaultConfigURL()
        self.config = ConfigStore.load(from: self.fileURL) ?? .default
    }

    func save() {
        do {
            let data = try JSONEncoder().encode(config)
            try data.write(to: fileURL, options: [.atomic])
        } catch {
            print("Config save failed: \(error)")
        }
    }

    func reload() {
        if let loaded = ConfigStore.load(from: fileURL) {
            config = loaded
        }
    }

    private static func load(from url: URL) -> SlurmHUDConfig? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(SlurmHUDConfig.self, from: data)
    }

    private static func defaultConfigURL() -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".slurmhud.json")
    }
}
