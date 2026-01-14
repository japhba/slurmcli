import Foundation

struct ClusterSnapshot: Decodable {
    let generated_at: String?
    let partitions: [Partition]
}

struct Partition: Decodable, Identifiable {
    let idx: Int?
    let partition: String?
    let states: [String]?
    let nodes: [String]?
    let configs: [PartitionConfig]?
    let node_details: [NodeDetail]?

    var id: String { partition ?? UUID().uuidString }
}

struct PartitionConfig: Decodable {
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

struct NodeDetail: Decodable, Identifiable {
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

struct JobDetail: Decodable {
    let name: String?
    let user: String?
    let time_info: String?
    let resource_summary: String?
}
