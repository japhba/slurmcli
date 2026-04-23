import Foundation

/// CLI entry point used by the LaunchAgent. Runs an SSH fetch, decodes the
/// snapshot, writes the shared cache, and exits. Always exits with status 0
/// (errors are written into the cache as an `error` field), so launchd does
/// not flag the agent as crashed.
enum RefreshCacheCommand {
    static let flag = "--refresh-cache"

    static func run() -> Int32 {
        log("--- refresh start (pid \(getpid()), uid \(getuid())) ---")
        let totalStart = Date()

        guard let config = ConfigStore.loadShared() else {
            let message = "no config found at \(SnapshotCache.cacheURL.deletingLastPathComponent().path)"
            log("FAIL: \(message)")
            SnapshotCache.saveError(message)
            log("--- refresh done in \(elapsed(since: totalStart)) (config-missing) ---")
            return 0
        }

        if config.useMockData {
            let snap = MockClusterData.snapshot()
            SnapshotCache.save(snapshot: snap)
            log("OK (mock): wrote \(snap.partitions.count) partition(s) in \(elapsed(since: totalStart))")
            return 0
        }

        let host = config.host.trimmingCharacters(in: .whitespacesAndNewlines)
        let command = config.command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !host.isEmpty, !command.isEmpty else {
            let message = "host or command is empty (host=\"\(host)\", command=\"\(command)\")"
            log("FAIL: \(message)")
            SnapshotCache.saveError(message)
            return 0
        }

        let timeout = max(3, config.timeoutSeconds)
        log("ssh \(host) \(command)  (timeout=\(timeout)s)")

        let runner = SSHRunner(host: host, command: command, timeoutSeconds: timeout)
        let sshStart = Date()
        let data: Data
        do {
            data = try runner.run()
        } catch let error as SSHRunnerError {
            let message = describe(error)
            log("FAIL ssh after \(elapsed(since: sshStart)): \(message)")
            SnapshotCache.saveError(message)
            log("--- refresh done in \(elapsed(since: totalStart)) (ssh-fail) ---")
            return 0
        } catch {
            let message = "ssh: \(error)"
            log("FAIL ssh after \(elapsed(since: sshStart)): \(message)")
            SnapshotCache.saveError(message)
            log("--- refresh done in \(elapsed(since: totalStart)) (ssh-fail) ---")
            return 0
        }
        log("ssh ok: \(data.count) bytes in \(elapsed(since: sshStart))")

        let parseStart = Date()
        let snapshot: ClusterSnapshot
        do {
            snapshot = try JSONDecoder().decode(ClusterSnapshot.self, from: data)
        } catch {
            let preview = String(data: data.prefix(200), encoding: .utf8) ?? "<non-utf8>"
            let message = "decode failed: \(error). Output starts with: \(preview)"
            log("FAIL parse: \(message)")
            SnapshotCache.saveError(message)
            log("--- refresh done in \(elapsed(since: totalStart)) (parse-fail) ---")
            return 0
        }
        log("parse ok: \(snapshot.partitions.count) partition(s) in \(elapsed(since: parseStart))")

        SnapshotCache.save(snapshot: snapshot)
        log("OK: wrote cache in \(elapsed(since: totalStart)) -> \(SnapshotCache.cacheURL.path)")
        return 0
    }

    // MARK: - Logging

    /// Append a single timestamped line to stderr. When invoked by launchd
    /// via `BackgroundRefreshAgent`, stderr is redirected to
    /// `~/Library/Application Support/SlurmHUD/refresh.log` via the
    /// plist's `StandardErrorPath`. When invoked manually from a terminal
    /// for debugging, the line lands on the terminal instead — redirect
    /// as you see fit.
    private static func log(_ message: String) {
        let line = "[\(timestamp())] \(message)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private static let timestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        f.timeZone = TimeZone.current
        return f
    }()

    private static func timestamp() -> String {
        timestampFormatter.string(from: Date())
    }

    private static func elapsed(since start: Date) -> String {
        String(format: "%.2fs", Date().timeIntervalSince(start))
    }

    private static func describe(_ error: SSHRunnerError) -> String {
        switch error {
        case .failedToLaunch:
            return "failed to launch /usr/bin/ssh"
        case .timeout(let elapsed, let stderr):
            let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                return String(format: "ssh wall-clock timeout after %.1fs", elapsed)
            }
            return String(format: "ssh wall-clock timeout after %.1fs: %@", elapsed, trimmed)
        case .nonZeroExit(let code, let stderr):
            let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                return "ssh exited with status \(code)"
            }
            return "ssh exited with status \(code): \(trimmed)"
        }
    }
}
