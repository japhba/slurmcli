import Foundation

final class StatusFetcher: ObservableObject {
    @Published var snapshot: ClusterSnapshot?
    @Published var lastError: String?

    private let decoder = JSONDecoder()

    func fetch(config: SlurmHUDConfig, completion: ((ClusterSnapshot?) -> Void)? = nil) {
        DispatchQueue.global().async {
            let runner = SSHRunner(
                host: config.host,
                command: config.command,
                timeoutSeconds: config.timeoutSeconds
            )
            do {
                let data = try runner.run()
                let snapshot = try self.decoder.decode(ClusterSnapshot.self, from: data)
                DispatchQueue.main.async {
                    self.snapshot = snapshot
                    self.lastError = nil
                    // Cache snapshot for widget access
                    SnapshotCache.save(snapshot: snapshot)
                    completion?(snapshot)
                }
            } catch {
                DispatchQueue.main.async {
                    self.lastError = "\(error)"
                    // Cache error for widget
                    SnapshotCache.saveError("\(error)")
                    completion?(nil)
                }
            }
        }
    }
}
