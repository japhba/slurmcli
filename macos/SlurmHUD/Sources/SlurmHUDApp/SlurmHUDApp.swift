import SwiftUI

@main
struct SlurmHUDApp: App {
    @StateObject private var configStore = ConfigStore()
    @StateObject private var fetcher = StatusFetcher()

    var body: some Scene {
        WindowGroup {
            StatusView()
                .environmentObject(configStore)
                .environmentObject(fetcher)
        }
        .defaultSize(width: 900, height: 600)

        Settings {
            SettingsView()
                .environmentObject(configStore)
        }
    }
}
