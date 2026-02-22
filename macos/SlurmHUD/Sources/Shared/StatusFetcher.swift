import Foundation

struct LogEntry: Identifiable {
    let id = UUID()
    let date: Date
    let message: String
}

final class StatusFetcher: ObservableObject {
    @Published var snapshot: ClusterSnapshot?
    @Published var lastError: String?
    @Published var log: [LogEntry] = []

    private let decoder = JSONDecoder()
    private static let maxLogEntries = 50

    private func appendLog(_ message: String) {
        let entry = LogEntry(date: Date(), message: message)
        DispatchQueue.main.async {
            self.log.append(entry)
            if self.log.count > Self.maxLogEntries {
                self.log.removeFirst(self.log.count - Self.maxLogEntries)
            }
        }
    }

    func fetch(config: SlurmHUDConfig, completion: ((ClusterSnapshot?) -> Void)? = nil) {
        appendLog("Fetching from \(config.host)...")
        DispatchQueue.global().async {
            let runner = SSHRunner(
                host: config.host,
                command: config.command,
                timeoutSeconds: config.timeoutSeconds
            )
            do {
                let data = try runner.run()
                let snapshot = try self.decoder.decode(ClusterSnapshot.self, from: data)
                self.appendLog("OK – \(snapshot.partitions.count) partition(s)")
                DispatchQueue.main.async {
                    self.snapshot = snapshot
                    self.lastError = nil
                    SnapshotCache.save(snapshot: snapshot)
                    completion?(snapshot)
                }
            } catch {
                self.appendLog("Error: \(error)")
                DispatchQueue.main.async {
                    self.lastError = "\(error)"
                    SnapshotCache.saveError("\(error)")
                    completion?(nil)
                }
            }
        }
    }
}
