import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var configStore: ConfigStore

    @State private var host: String = ""
    @State private var refreshSeconds: String = ""
    @State private var timeoutSeconds: String = ""
    @State private var command: String = ""

    var body: some View {
        Form {
            TextField("SSH host", text: $host)
            TextField("Refresh seconds", text: $refreshSeconds)
            TextField("Timeout seconds", text: $timeoutSeconds)
            TextField("Command", text: $command)
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
        .frame(width: 360)
        .onAppear {
            syncFromStore()
        }
    }

    private func syncFromStore() {
        host = configStore.config.host
        refreshSeconds = "\(configStore.config.refreshSeconds)"
        timeoutSeconds = "\(configStore.config.timeoutSeconds)"
        command = configStore.config.command
    }

    private func save() {
        let refresh = Int(refreshSeconds) ?? configStore.config.refreshSeconds
        let timeout = Int(timeoutSeconds) ?? configStore.config.timeoutSeconds
        configStore.config = SlurmHUDConfig(
            host: host.isEmpty ? configStore.config.host : host,
            refreshSeconds: max(30, refresh),
            timeoutSeconds: max(3, timeout),
            command: command.isEmpty ? configStore.config.command : command
        )
        configStore.save()
    }
}
