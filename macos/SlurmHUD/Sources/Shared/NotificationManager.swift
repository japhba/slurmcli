import Foundation
import UserNotifications

final class NotificationManager {
    static let shared = NotificationManager()

    func requestPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if let error = error {
                print("Notification permission error: \(error)")
            }
        }
    }

    func notifyJobFinished(jobName: String, user: String, node: String, elapsed: String?) {
        let content = UNMutableNotificationContent()
        content.title = "Job finished on \(node)"
        var body = "\(jobName)"
        if !user.isEmpty { body += " (\(user))" }
        if let elapsed = elapsed { body += " — ran for \(elapsed)" }
        content.body = body
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )
        UNUserNotificationCenter.current().add(request)
    }

    func detectFinishedJobs(
        oldSnapshot: ClusterSnapshot?,
        newSnapshot: ClusterSnapshot,
        watchedNodes: Set<String>
    ) {
        guard let oldSnapshot = oldSnapshot else { return }
        if watchedNodes.isEmpty { return }

        // Build lookup of old jobs per watched node
        var oldJobs: [String: Set<Int>] = [:]
        for partition in oldSnapshot.partitions {
            guard let nodes = partition.node_details else { continue }
            for node in nodes {
                guard let name = node.node, watchedNodes.contains(name) else { continue }
                guard let jobs = node.jobs else { continue }
                let ids = Set(jobs.compactMap { $0.id })
                oldJobs[name, default: []].formUnion(ids)
            }
        }

        // Compare with new snapshot
        for partition in newSnapshot.partitions {
            guard let nodes = partition.node_details else { continue }
            for node in nodes {
                guard let name = node.node, watchedNodes.contains(name) else { continue }
                let newIDs = Set((node.jobs ?? []).compactMap { $0.id })
                guard let oldIDs = oldJobs[name] else { continue }

                let finished = oldIDs.subtracting(newIDs)
                if finished.isEmpty { continue }

                // Look up job details from old snapshot
                let oldNodeJobs = oldSnapshot.partitions
                    .flatMap { $0.node_details ?? [] }
                    .filter { $0.node == name }
                    .flatMap { $0.jobs ?? [] }

                for jobID in finished {
                    if let job = oldNodeJobs.first(where: { $0.id == jobID }) {
                        notifyJobFinished(
                            jobName: job.name ?? "Job #\(jobID)",
                            user: job.user ?? "",
                            node: name,
                            elapsed: job.elapsed
                        )
                    } else {
                        notifyJobFinished(
                            jobName: "Job #\(jobID)",
                            user: "",
                            node: name,
                            elapsed: nil
                        )
                    }
                }
            }
        }
    }
}
