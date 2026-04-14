import Foundation
#if canImport(WidgetKit)
import WidgetKit
#endif

struct LogEntry: Identifiable {
    let id = UUID()
    let date: Date
    let message: String
}

final class StatusFetcher: ObservableObject {
    @Published var snapshot: ClusterSnapshot?
    @Published var lastError: String?
    @Published var log: [LogEntry] = []
    @Published var isFetching = false

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

    /// Called on main thread after a successful fetch with (oldSnapshot, newSnapshot).
    var onSnapshotUpdate: ((ClusterSnapshot?, ClusterSnapshot) -> Void)?

    private func reloadWidgetTimelines() {
        #if canImport(WidgetKit)
        WidgetCenter.shared.reloadAllTimelines()
        #endif
    }

    func fetch(config: SlurmHUDConfig, completion: ((ClusterSnapshot?) -> Void)? = nil) {
        var shouldStart = true
        let beginFetch = {
            if self.isFetching {
                shouldStart = false
            } else {
                self.isFetching = true
            }
        }

        if Thread.isMainThread {
            beginFetch()
        } else {
            DispatchQueue.main.sync(execute: beginFetch)
        }

        guard shouldStart else {
            completion?(snapshot)
            return
        }

        appendLog(config.useMockData ? "Loading bundled dummy data..." : "Fetching from \(config.host)...")
        DispatchQueue.global().async {
            if config.useMockData {
                let newSnapshot = MockClusterData.snapshot()
                self.appendLog("OK – dummy snapshot with \(newSnapshot.partitions.count) partition(s)")
                DispatchQueue.main.async {
                    let oldSnapshot = self.snapshot
                    self.snapshot = newSnapshot
                    self.lastError = nil
                    self.isFetching = false
                    SnapshotCache.save(snapshot: newSnapshot)
                    self.reloadWidgetTimelines()
                    self.onSnapshotUpdate?(oldSnapshot, newSnapshot)
                    completion?(newSnapshot)
                }
                return
            }

            let runner = SSHRunner(
                host: config.host,
                command: config.command,
                timeoutSeconds: config.timeoutSeconds
            )
            do {
                let data = try runner.run()
                let newSnapshot = try self.decoder.decode(ClusterSnapshot.self, from: data)
                self.appendLog("OK – \(newSnapshot.partitions.count) partition(s)")
                DispatchQueue.main.async {
                    let oldSnapshot = self.snapshot
                    self.snapshot = newSnapshot
                    self.lastError = nil
                    self.isFetching = false
                    SnapshotCache.save(snapshot: newSnapshot)
                    self.reloadWidgetTimelines()
                    self.onSnapshotUpdate?(oldSnapshot, newSnapshot)
                    completion?(newSnapshot)
                }
            } catch {
                self.appendLog("Error: \(error)")
                DispatchQueue.main.async {
                    self.isFetching = false
                    self.lastError = "\(error)"
                    SnapshotCache.saveError("\(error)")
                    self.reloadWidgetTimelines()
                    completion?(nil)
                }
            }
        }
    }
}
