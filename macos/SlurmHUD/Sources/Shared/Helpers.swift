import Foundation

enum HUDHelpers {
    static func formatJobSummary(_ jobs: [JobDetail]?) -> String {
        guard let jobs, !jobs.isEmpty else { return "no jobs" }
        let first = jobs[0]
        let name = first.name ?? "job"
        let user = first.user ?? ""
        if user.isEmpty {
            return "\(jobs.count) job(s): \(name)"
        }
        return "\(jobs.count) job(s): \(name) (\(user))"
    }

    static func formatGPUType(_ node: NodeDetail) -> String {
        let label = node.gpu_label_text ?? ""
        return label.isEmpty ? "GPU: n/a" : "GPU: \(label)"
    }
}
