import Foundation

enum SSHRunnerError: Error {
    case failedToLaunch
    case timeout
    case nonZeroExit(code: Int, stderr: String)
}

struct SSHRunner {
    let host: String
    let command: String
    let timeoutSeconds: Int

    func run() throws -> Data {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")
        process.arguments = [
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=\(timeoutSeconds)",
            host,
            command
        ]

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        try process.run()

        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global().async {
            process.waitUntilExit()
            group.leave()
        }

        let waitResult = group.wait(timeout: .now() + .seconds(timeoutSeconds))
        if waitResult == .timedOut {
            process.terminate()
            throw SSHRunnerError.timeout
        }

        let status = process.terminationStatus
        if status != 0 {
            let errData = stderr.fileHandleForReading.readDataToEndOfFile()
            let errText = String(data: errData, encoding: .utf8) ?? ""
            throw SSHRunnerError.nonZeroExit(code: Int(status), stderr: errText)
        }

        return stdout.fileHandleForReading.readDataToEndOfFile()
    }
}
