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
                }
        }
        .defaultSize(width: 900, height: 600)

        Settings {
            SettingsView()
                .environmentObject(configStore)
                .environmentObject(fetcher)
        }
    }
}
