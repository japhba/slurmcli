import SwiftUI

@main
struct SlurmHUDApp: App {
    @StateObject private var configStore = ConfigStore()
    @StateObject private var fetcher = StatusFetcher()
    @StateObject private var watchStore = WatchStore()

    var body: some Scene {
        WindowGroup {
            StatusView()
                .environmentObject(configStore)
                .environmentObject(fetcher)
                .environmentObject(watchStore)
                .onAppear {
                    fetcher.onSnapshotUpdate = { old, new in
                        NotificationManager.shared.detectFinishedJobs(
                            oldSnapshot: old,
                            newSnapshot: new,
                            watchedNodes: watchStore.watchedNodes
                        )
                    }
                    NotificationManager.shared.requestPermission()
                    syncBackgroundRefresh(config: configStore.config)
                }
                .onReceive(configStore.$config) { newConfig in
                    syncBackgroundRefresh(config: newConfig)
                }
        }
        .defaultSize(width: 900, height: 600)

        Settings {
            SettingsView()
                .environmentObject(configStore)
                .environmentObject(fetcher)
        }
    }

    private func syncBackgroundRefresh(config: SlurmHUDConfig) {
        if config.backgroundRefreshEnabled {
            BackgroundRefreshAgent.install(config: config)
        } else {
            BackgroundRefreshAgent.uninstall()
        }
    }
}
