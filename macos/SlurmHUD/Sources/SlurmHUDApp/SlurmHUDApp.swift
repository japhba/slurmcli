import SwiftUI

@main
struct SlurmHUDApp: App {
    @StateObject private var configStore = ConfigStore()
    @StateObject private var fetcher = StatusFetcher()

    var body: some Scene {
        MenuBarExtra("SlurmHUD", systemImage: "chart.bar.xaxis") {
            StatusView()
                .environmentObject(configStore)
                .environmentObject(fetcher)
        }
        .menuBarExtraStyle(.window)

        Settings {
            SettingsView()
                .environmentObject(configStore)
        }
    }
}
