import Foundation

/// Manages a per-user LaunchAgent that periodically invokes the SlurmHUD
/// binary in `--refresh-cache` mode. Running the entitled SlurmHUD binary
/// (rather than a plain bash script) is what lets the background refresh
/// actually write into the App Group container — TCC blocks unsandboxed
/// launchd-spawned bash from touching `~/Library/Group Containers/...`,
/// which used to cause every refresh to silently fail with "Operation not
/// permitted" while still logging "refresh ok".
enum BackgroundRefreshAgent {
    static let label = "com.jbauer.slurmhud.refresh"

    private static var supportDirectory: URL {
        FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support", isDirectory: true)
            .appendingPathComponent("SlurmHUD", isDirectory: true)
    }

    private static var logURL: URL {
        supportDirectory.appendingPathComponent("refresh.log", isDirectory: false)
    }

    /// Path to the legacy bash script — left here so we can clean it up on
    /// install/uninstall after the cutover to the binary-based agent.
    private static var legacyScriptURL: URL {
        supportDirectory.appendingPathComponent("refresh-cache.sh", isDirectory: false)
    }

    private static var plistURL: URL {
        FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent("Library/LaunchAgents", isDirectory: true)
            .appendingPathComponent("\(label).plist", isDirectory: false)
    }

    /// Install (or update) the LaunchAgent so it fires every
    /// `config.refreshSeconds`. Skips installation when running with mock
    /// data or when the host/command are not configured.
    @discardableResult
    static func install(config: SlurmHUDConfig) -> Bool {
        guard !config.useMockData,
              !config.host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              !config.command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            uninstall()
            return false
        }

        do {
            try writePlist(config: config)
        } catch {
            NSLog("BackgroundRefreshAgent: failed to write plist: \(error)")
            return false
        }

        // Drop the old bash script — it has been replaced by the binary.
        try? FileManager.default.removeItem(at: legacyScriptURL)

        let target = "gui/\(getuid())"
        runLaunchctl(["bootout", "\(target)/\(label)"], expectSuccess: false)
        let bootstrapped = runLaunchctl(["bootstrap", target, plistURL.path], expectSuccess: true)
        if !bootstrapped {
            _ = runLaunchctl(["load", "-w", plistURL.path], expectSuccess: true)
        }
        return true
    }

    /// Remove the LaunchAgent.
    static func uninstall() {
        let target = "gui/\(getuid())/\(label)"
        runLaunchctl(["bootout", target], expectSuccess: false)
        if FileManager.default.fileExists(atPath: plistURL.path) {
            runLaunchctl(["unload", "-w", plistURL.path], expectSuccess: false)
        }
        try? FileManager.default.removeItem(at: plistURL)
    }

    // MARK: - Internals

    /// Path to the currently-running SlurmHUD binary. The LaunchAgent
    /// re-invokes this same binary with `--refresh-cache`. Reinstalling is
    /// idempotent and we re-install on every app launch (see
    /// `SlurmHUDApp.syncBackgroundRefresh`), so a moved app picks up the
    /// new path next time the user opens SlurmHUD.
    private static func executablePath() -> String {
        if let url = Bundle.main.executableURL {
            return url.path
        }
        return CommandLine.arguments.first ?? "/usr/bin/false"
    }

    private static func writePlist(config: SlurmHUDConfig) throws {
        let interval = max(60, config.refreshSeconds)

        try FileManager.default.createDirectory(
            at: supportDirectory,
            withIntermediateDirectories: true
        )

        let plist: [String: Any] = [
            "Label": label,
            "ProgramArguments": [
                executablePath(),
                RefreshCacheCommand.flag,
            ],
            "StartInterval": interval,
            "RunAtLoad": true,
            "StandardOutPath": logURL.path,
            "StandardErrorPath": logURL.path,
            "EnvironmentVariables": [
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin",
                "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
            ],
            // Keep launchd from hammering us if the binary crashes early —
            // wait at least the configured interval between attempts.
            "ThrottleInterval": interval,
        ]

        try FileManager.default.createDirectory(
            at: plistURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = try PropertyListSerialization.data(
            fromPropertyList: plist,
            format: .xml,
            options: 0
        )
        try data.write(to: plistURL, options: .atomic)
    }

    @discardableResult
    private static func runLaunchctl(_ arguments: [String], expectSuccess: Bool) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = arguments
        let stderr = Pipe()
        process.standardError = stderr
        process.standardOutput = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            if expectSuccess {
                NSLog("BackgroundRefreshAgent: launchctl \(arguments.joined(separator: " ")) failed: \(error)")
            }
            return false
        }
        if process.terminationStatus != 0 && expectSuccess {
            let errData = stderr.fileHandleForReading.readDataToEndOfFile()
            let errText = String(data: errData, encoding: .utf8) ?? ""
            NSLog("BackgroundRefreshAgent: launchctl \(arguments.joined(separator: " ")) exited \(process.terminationStatus): \(errText)")
        }
        return process.terminationStatus == 0
    }
}
