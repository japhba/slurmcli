import Foundation

struct ClusterSnapshot: Codable {
    let generated_at: String?
    let partitions: [Partition]
}

struct Partition: Codable, Identifiable {
    let idx: Int?
    let partition: String?
    let states: [String]?
    let nodes: [String]?
    let configs: [PartitionConfig]?
    let node_details: [NodeDetail]?
    var pending_jobs: [PendingJob]? = nil

    var id: String { partition ?? UUID().uuidString }
}

struct PendingJob: Codable {
    let id: Int?
    let user: String?
    let name: String?
    let elapsed: String?
    let time_limit: String?
    let nodes_requested: String?
    let reason: String?

    private let _uuid = UUID()

    private enum CodingKeys: String, CodingKey {
        case id, user, name, elapsed, time_limit, nodes_requested, reason
    }

    var jobIdentifier: String {
        id.map(String.init) ?? _uuid.uuidString
    }
}

struct PartitionConfig: Codable {
    let mem_gb: Int?
    let cpus: Int?
    let gpus: Int?
    let node_count: Int?
    let available_count: Int?
    let gpu_in_use: Int?
    let gpu_total: Int?
    let users_min: Int?
    let users_max: Int?
}

struct NodeDetail: Codable, Identifiable {
    let node: String?
    let state: String?
    let memory_gb: Double?
    let cpus_alloc: Int?
    let cpus_total: Int?
    let gpus_in_use: Int?
    let gpus_total: Int?
    let gpu_label_text: String?
    let users: [String]?
    let jobs: [JobDetail]?

    var id: String { node ?? UUID().uuidString }
}

struct JobDetail: Codable, Identifiable {
    let id: Int?
    let name: String?
    let user: String?
    let elapsed: String?
    let time_left: String?
    let time_info: String?
    let resource_summary: String?
}
