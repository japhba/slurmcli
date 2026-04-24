import Foundation

enum MockClusterData {
    private static let timestampFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    static func snapshot(referenceDate: Date = Date()) -> ClusterSnapshot {
        ClusterSnapshot(
            generated_at: timestampFormatter.string(from: referenceDate),
            partitions: [
                Partition(
                    idx: 0,
                    partition: "gpu-a100",
                    states: ["mix", "alloc"],
                    nodes: ["gpu001", "gpu002", "gpu003", "gpu004"],
                    configs: [
                        PartitionConfig(
                            mem_gb: 512,
                            cpus: 64,
                            gpus: 8,
                            node_count: 4,
                            available_count: 1,
                            gpu_in_use: 25,
                            gpu_total: 32,
                            users_min: nil,
                            users_max: nil
                        )
                    ],
                    node_details: [
                        NodeDetail(
                            node: "gpu001",
                            state: "alloc",
                            memory_gb: 512,
                            tmp_disk_gb: 1024,
                            cpus_alloc: 64,
                            cpus_total: 64,
                            gpus_in_use: 8,
                            gpus_total: 8,
                            gpu_label_text: "A100",
                            users: ["alice"],
                            jobs: [
                                JobDetail(
                                    id: 101,
                                    name: "train-diffusion",
                                    user: "alice",
                                    elapsed: "02:14:00",
                                    time_left: "05:46:00",
                                    time_info: nil,
                                    resource_summary: "cpu=64, gpu=8; mem=480G"
                                )
                            ]
                        ),
                        NodeDetail(
                            node: "gpu002",
                            state: "mix",
                            memory_gb: 512,
                            tmp_disk_gb: 1024,
                            cpus_alloc: 32,
                            cpus_total: 64,
                            gpus_in_use: 5,
                            gpus_total: 8,
                            gpu_label_text: "A100",
                            users: ["bob"],
                            jobs: [
                                JobDetail(
                                    id: 102,
                                    name: "finetune-llm",
                                    user: "bob",
                                    elapsed: "00:48:00",
                                    time_left: "03:12:00",
                                    time_info: nil,
                                    resource_summary: "cpu=32, gpu=5; mem=256G"
                                )
                            ]
                        ),
                        NodeDetail(
                            node: "gpu003",
                            state: "idle",
                            memory_gb: 512,
                            tmp_disk_gb: 1024,
                            cpus_alloc: 0,
                            cpus_total: 64,
                            gpus_in_use: 0,
                            gpus_total: 8,
                            gpu_label_text: "A100",
                            users: [],
                            jobs: []
                        ),
                        NodeDetail(
                            node: "gpu004",
                            state: "alloc",
                            memory_gb: 512,
                            tmp_disk_gb: 1024,
                            cpus_alloc: 64,
                            cpus_total: 64,
                            gpus_in_use: 8,
                            gpus_total: 8,
                            gpu_label_text: "A100",
                            users: ["carol"],
                            jobs: [
                                JobDetail(
                                    id: 103,
                                    name: "render",
                                    user: "carol",
                                    elapsed: "03:03:00",
                                    time_left: "00:57:00",
                                    time_info: nil,
                                    resource_summary: "cpu=64, gpu=8; mem=320G"
                                )
                            ]
                        )
                    ],
                    pending_jobs: [
                        PendingJob(
                            id: 901,
                            user: "eve",
                            name: "hpo-sweep",
                            elapsed: "00:14:00",
                            time_limit: "08:00:00",
                            nodes_requested: "2",
                            reason: "Resources"
                        ),
                        PendingJob(
                            id: 902,
                            user: "frank",
                            name: "eval",
                            elapsed: "00:02:00",
                            time_limit: "01:00:00",
                            nodes_requested: "1",
                            reason: "Priority"
                        )
                    ]
                ),
                Partition(
                    idx: 1,
                    partition: "cpu-fast",
                    states: ["idle"],
                    nodes: ["cpu001", "cpu002", "cpu003", "cpu004", "cpu005", "cpu006"],
                    configs: [
                        PartitionConfig(
                            mem_gb: 256,
                            cpus: 96,
                            gpus: 0,
                            node_count: 6,
                            available_count: 4,
                            gpu_in_use: 0,
                            gpu_total: 0,
                            users_min: nil,
                            users_max: nil
                        )
                    ],
                    node_details: [
                        NodeDetail(
                            node: "cpu001",
                            state: "idle",
                            memory_gb: 256,
                            tmp_disk_gb: 1024,
                            cpus_alloc: 0,
                            cpus_total: 96,
                            gpus_in_use: 0,
                            gpus_total: 0,
                            gpu_label_text: nil,
                            users: [],
                            jobs: []
                        ),
                        NodeDetail(
                            node: "cpu002",
                            state: "idle",
                            memory_gb: 256,
                            tmp_disk_gb: 1024,
                            cpus_alloc: 0,
                            cpus_total: 96,
                            gpus_in_use: 0,
                            gpus_total: 0,
                            gpu_label_text: nil,
                            users: [],
                            jobs: []
                        )
                    ]
                ),
                Partition(
                    idx: 2,
                    partition: "long",
                    states: ["drain"],
                    nodes: ["long001", "long002"],
                    configs: [
                        PartitionConfig(
                            mem_gb: 512,
                            cpus: 64,
                            gpus: 0,
                            node_count: 2,
                            available_count: 0,
                            gpu_in_use: 0,
                            gpu_total: 0,
                            users_min: nil,
                            users_max: nil
                        )
                    ],
                    node_details: [
                        NodeDetail(
                            node: "long001",
                            state: "drain",
                            memory_gb: 512,
                            tmp_disk_gb: 1024,
                            cpus_alloc: 0,
                            cpus_total: 64,
                            gpus_in_use: 0,
                            gpus_total: 0,
                            gpu_label_text: nil,
                            users: [],
                            jobs: []
                        ),
                        NodeDetail(
                            node: "long002",
                            state: "alloc",
                            memory_gb: 512,
                            tmp_disk_gb: 1024,
                            cpus_alloc: 64,
                            cpus_total: 64,
                            gpus_in_use: 0,
                            gpus_total: 0,
                            gpu_label_text: nil,
                            users: ["dave"],
                            jobs: [
                                JobDetail(
                                    id: 201,
                                    name: "sim",
                                    user: "dave",
                                    elapsed: "12:21:00",
                                    time_left: "11:39:00",
                                    time_info: nil,
                                    resource_summary: "cpu=64; mem=384G"
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    }
}
