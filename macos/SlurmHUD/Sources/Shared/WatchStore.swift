import Foundation

final class WatchStore: ObservableObject {
    private static let key = "watchedNodes"

    @Published var watchedNodes: Set<String> {
        didSet { save() }
    }

    init() {
        let saved = UserDefaults.standard.stringArray(forKey: Self.key) ?? []
        self.watchedNodes = Set(saved)
    }

    func isWatched(_ node: String) -> Bool {
        watchedNodes.contains(node)
    }

    func toggle(_ node: String) {
        if watchedNodes.contains(node) {
            watchedNodes.remove(node)
        } else {
            watchedNodes.insert(node)
        }
    }

    private func save() {
        UserDefaults.standard.set(Array(watchedNodes), forKey: Self.key)
    }
}
