import SwiftUI
#if canImport(WidgetKit)
import WidgetKit
#endif

struct SettingsView: View {
    @EnvironmentObject var configStore: ConfigStore
    @EnvironmentObject var fetcher: StatusFetcher

    @State private var host: String = ""
    @State private var refreshSeconds: String = ""
    @State private var timeoutSeconds: String = ""
    @State private var command: String = ""
    @State private var widgetPartition: String = ""
    @State private var useMockData = false

    var body: some View {
        Form {
            Toggle("Use Dummy Data", isOn: $useMockData)
            TextField("SSH host", text: $host)
                .disabled(useMockData)
            TextField("Refresh seconds", text: $refreshSeconds)
            TextField("Timeout seconds", text: $timeoutSeconds)
            TextField("Command", text: $command)
                .disabled(useMockData)
            TextField("Widget partition", text: $widgetPartition)
            Text("Leave blank to show all partitions in the widget.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if useMockData {
                Text("When enabled, SlurmHUD skips SSH and uses bundled sample cluster data for both the app and widget.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack {
                Button("Reload") {
                    configStore.reload()
                    syncFromStore()
                }
                Spacer()
                Button("Save") {
                    save()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(16)
        .frame(minWidth: 300, idealWidth: 400)
        .onAppear {
            syncFromStore()
        }
    }

    private func syncFromStore() {
        host = configStore.config.host
        refreshSeconds = "\(configStore.config.refreshSeconds)"
        timeoutSeconds = "\(configStore.config.timeoutSeconds)"
        command = configStore.config.command
        widgetPartition = configStore.config.widgetPartition ?? ""
        useMockData = configStore.config.useMockData
    }

    private func save() {
        let wasUsingMockData = configStore.config.useMockData
        let refresh = Int(refreshSeconds) ?? configStore.config.refreshSeconds
        let timeout = Int(timeoutSeconds) ?? configStore.config.timeoutSeconds
        configStore.config = SlurmHUDConfig(
            host: host.isEmpty ? configStore.config.host : host,
            refreshSeconds: max(30, refresh),
            timeoutSeconds: max(3, timeout),
            command: command.isEmpty ? configStore.config.command : command,
            widgetPartition: widgetPartition,
            useMockData: useMockData
        )
        configStore.save()

        if useMockData {
            let mockSnapshot = MockClusterData.snapshot()
            fetcher.snapshot = mockSnapshot
            fetcher.lastError = nil
            SnapshotCache.save(snapshot: mockSnapshot)
        } else if wasUsingMockData {
            let waitingMessage = "Waiting for the next scheduled refresh."
            fetcher.snapshot = nil
            fetcher.lastError = waitingMessage
            SnapshotCache.saveError(waitingMessage)
        }

        #if canImport(WidgetKit)
        WidgetCenter.shared.reloadAllTimelines()
        #endif
    }
}
