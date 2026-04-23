import Foundation

enum SSHRunnerError: Error {
    case failedToLaunch
    /// Total wall-clock time exceeded `timeoutSeconds`. `elapsed` is what we
    /// actually waited; `stderr` is anything ssh wrote before we killed it
    /// (often "kex_exchange_identification" / "Connection timed out").
    case timeout(elapsed: Double, stderr: String)
    case nonZeroExit(code: Int, stderr: String)
}

struct SSHRunner {
    let host: String
    let command: String
    /// Total wall-clock budget for ssh+command, in seconds. We use a tight
    /// `ConnectTimeout` (capped to 30s) so we fail fast on unreachable
    /// hosts instead of burning the whole budget on TCP handshake retries.
    let timeoutSeconds: Int

    private var connectTimeoutSeconds: Int {
        max(5, min(30, timeoutSeconds))
    }

    func run() throws -> Data {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")
        process.arguments = [
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=\(connectTimeoutSeconds)",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            host,
            command,
        ]

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        do {
            try process.run()
        } catch {
            throw SSHRunnerError.failedToLaunch
        }

        let start = Date()

        // Stream stderr/stdout off the pipe buffers so a chatty ssh
        // (e.g. -vvvv on the remote command) cannot deadlock us by
        // filling the 16K pipe buffer while we wait.
        let stdoutBuffer = StreamingBuffer(handle: stdout.fileHandleForReading)
        let stderrBuffer = StreamingBuffer(handle: stderr.fileHandleForReading)

        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global().async {
            process.waitUntilExit()
            group.leave()
        }

        let waitResult = group.wait(timeout: .now() + .seconds(timeoutSeconds))
        if waitResult == .timedOut {
            // SIGTERM, then a moment, then SIGKILL — terminate() alone can
            // leave a wedged ssh hanging if the remote stopped reading.
            process.terminate()
            _ = group.wait(timeout: .now() + .seconds(2))
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
                _ = group.wait(timeout: .now() + .seconds(1))
            }
            _ = stdoutBuffer.stop()
            let errText = stderrBuffer.stopAsString()
            throw SSHRunnerError.timeout(elapsed: Date().timeIntervalSince(start), stderr: errText)
        }

        let stdoutData = stdoutBuffer.stop()
        let stderrText = stderrBuffer.stopAsString()

        let status = process.terminationStatus
        if status != 0 {
            throw SSHRunnerError.nonZeroExit(code: Int(status), stderr: stderrText)
        }
        return stdoutData
    }
}

/// Drains a FileHandle on a background queue so the producer (ssh) never
/// blocks on a full pipe. `stop()` returns whatever has accumulated so far
/// and tears down the reader.
private final class StreamingBuffer: @unchecked Sendable {
    private let handle: FileHandle
    private let queue: DispatchQueue
    private var data = Data()
    private var stopped = false
    private let lock = NSLock()

    init(handle: FileHandle) {
        self.handle = handle
        self.queue = DispatchQueue(label: "SSHRunner.StreamingBuffer")
        queue.async { [weak self] in
            guard let self else { return }
            while true {
                let chunk = handle.availableData
                if chunk.isEmpty {
                    return
                }
                self.lock.lock()
                if self.stopped {
                    self.lock.unlock()
                    return
                }
                self.data.append(chunk)
                self.lock.unlock()
            }
        }
    }

    @discardableResult
    func stop() -> Data {
        lock.lock()
        if stopped {
            let snapshot = data
            lock.unlock()
            return snapshot
        }
        stopped = true
        lock.unlock()
        // Give the reader a brief moment to flush the last chunk.
        try? handle.close()
        lock.lock()
        let snapshot = data
        lock.unlock()
        return snapshot
    }
}

extension StreamingBuffer {
    func stopAsString() -> String {
        String(data: stop(), encoding: .utf8) ?? ""
    }
}
