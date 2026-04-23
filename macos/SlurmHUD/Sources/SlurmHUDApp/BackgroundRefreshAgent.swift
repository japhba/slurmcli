import Foundation

/// Manages a per-user LaunchAgent that periodically runs an SSH fetch and
/// writes the shared snapshot cache, so the widget stays current even when
/// the SlurmHUD app is not running.
///
/// The agent shells out to a small embedded script (`refresh-cache.sh`) that
/// runs `ssh <host> <command>` and wraps the JSON output in the same
/// `CachedSnapshot` shape that `SnapshotCache` decodes.
enum BackgroundRefreshAgent {
    static let label = "com.jbauer.slurmhud.refresh"

    private static var supportDirectory: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home
            .appendingPathComponent("Library/Application Support", isDirectory: true)
            .appendingPathComponent("SlurmHUD", isDirectory: true)
    }

    private static var scriptURL: URL {
        supportDirectory.appendingPathComponent("refresh-cache.sh", isDirectory: false)
    }

    private static var logURL: URL {
        supportDirectory.appendingPathComponent("refresh.log", isDirectory: false)
    }

    private static var plistURL: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home
            .appendingPathComponent("Library/LaunchAgents", isDirectory: true)
            .appendingPathComponent("\(label).plist", isDirectory: false)
    }

    /// Install (or update) the LaunchAgent so it fires every
    /// `config.refreshSeconds`. Skips installation when running with mock data
    /// or when the host/command are not configured.
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
            try writeScript()
            try writePlist(config: config)
        } catch {
            NSLog("BackgroundRefreshAgent: failed to write files: \(error)")
            return false
        }

        // Reload: bootout (ignore failure if not loaded), then bootstrap.
        let target = "gui/\(getuid())"
        runLaunchctl(["bootout", "\(target)/\(label)"], expectSuccess: false)
        let bootstrapped = runLaunchctl(["bootstrap", target, plistURL.path], expectSuccess: true)
        if !bootstrapped {
            // Fall back to legacy load -w in case bootstrap is unavailable.
            _ = runLaunchctl(["load", "-w", plistURL.path], expectSuccess: true)
        }
        return true
    }

    /// Remove the LaunchAgent.
    static func uninstall() {
        let target = "gui/\(getuid())/\(label)"
        runLaunchctl(["bootout", target], expectSuccess: false)
        // Legacy fallback in case bootout failed because the agent was loaded
        // via `load -w` historically.
        if FileManager.default.fileExists(atPath: plistURL.path) {
            runLaunchctl(["unload", "-w", plistURL.path], expectSuccess: false)
        }
        try? FileManager.default.removeItem(at: plistURL)
    }

    // MARK: - Internals

    private static func writeScript() throws {
        try FileManager.default.createDirectory(
            at: supportDirectory,
            withIntermediateDirectories: true
        )
        let data = Data(scriptContents.utf8)
        let existing = try? Data(contentsOf: scriptURL)
        if existing != data {
            try data.write(to: scriptURL, options: .atomic)
        }
        // Ensure the script is executable.
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: scriptURL.path
        )
    }

    private static func writePlist(config: SlurmHUDConfig) throws {
        let cachePath = SnapshotCache.cacheURL.path
        let interval = max(60, config.refreshSeconds)
        let timeoutString = String(max(3, config.timeoutSeconds))

        let plist: [String: Any] = [
            "Label": label,
            "ProgramArguments": [
                "/bin/bash",
                scriptURL.path,
                cachePath,
                config.host,
                config.command,
                timeoutString,
            ],
            "StartInterval": interval,
            "RunAtLoad": true,
            "StandardOutPath": logURL.path,
            "StandardErrorPath": logURL.path,
            // Hand the script a sane PATH so /usr/bin/ssh + helpers resolve.
            "EnvironmentVariables": [
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin",
                "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
            ],
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

    /// Embedded shell script. Reads the cache file path, host, command and
    /// timeout from positional arguments. Writes a JSON document matching
    /// Swift's default `JSONEncoder` encoding of `CachedSnapshot`, where
    /// `fetchedAt` is a `Double` of seconds since the Apple reference date
    /// (2001-01-01 00:00:00 UTC).
    private static let scriptContents: String = #"""
    #!/bin/bash
    set -u

    CACHE_FILE="${1:-}"
    HOST="${2:-}"
    COMMAND="${3:-}"
    TIMEOUT="${4:-180}"

    if [ -z "$CACHE_FILE" ] || [ -z "$HOST" ] || [ -z "$COMMAND" ]; then
        echo "[$(date)] usage error: missing arguments" >&2
        exit 64
    fi

    NOW_UNIX=$(date -u +%s)
    # Apple reference date offset from Unix epoch.
    FETCHED_AT=$(awk -v n="$NOW_UNIX" 'BEGIN { printf "%.6f", n - 978307200 }')

    mkdir -p "$(dirname "$CACHE_FILE")"

    TMP_OUT=$(mktemp -t slurmhud-refresh)
    TMP_ERR=$(mktemp -t slurmhud-refresh-err)
    trap 'rm -f "$TMP_OUT" "$TMP_ERR"' EXIT

    if /usr/bin/ssh \
        -o BatchMode=yes \
        -o ConnectTimeout="$TIMEOUT" \
        -o ServerAliveInterval=15 \
        "$HOST" "$COMMAND" >"$TMP_OUT" 2>"$TMP_ERR"
    then
        FIRST_CHAR=$(head -c1 "$TMP_OUT")
        if [ "$FIRST_CHAR" != "{" ]; then
            echo "[$(date)] ssh succeeded but output is not JSON" >&2
            cat "$TMP_ERR" >&2
            exit 65
        fi
        TMP_CACHE="${CACHE_FILE}.tmp"
        {
            printf '{"snapshot":'
            cat "$TMP_OUT"
            printf ',"fetchedAt":%s}\n' "$FETCHED_AT"
        } > "$TMP_CACHE" && mv "$TMP_CACHE" "$CACHE_FILE"
        echo "[$(date)] refresh ok ($(wc -c <"$TMP_OUT" | tr -d ' ') bytes)"
    else
        STATUS=$?
        echo "[$(date)] ssh failed with status $STATUS" >&2
        cat "$TMP_ERR" >&2
        TMP_CACHE="${CACHE_FILE}.tmp"
        {
            printf '{"snapshot":{"generated_at":null,"partitions":[]},"fetchedAt":%s,"error":"Background refresh failed (ssh status %s). See refresh.log."}\n' "$FETCHED_AT" "$STATUS"
        } > "$TMP_CACHE" && mv "$TMP_CACHE" "$CACHE_FILE"
        exit "$STATUS"
    fi
    """#
}
