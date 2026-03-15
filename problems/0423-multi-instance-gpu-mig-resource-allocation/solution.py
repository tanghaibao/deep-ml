def mig_resource_allocation(gpu_config: dict, mig_profiles: list, workloads: list) -> dict:
    """
    Allocate MIG instances to workloads on a single GPU.
    
    Args:
        gpu_config: Dict with 'total_compute_slices' and 'total_memory_gb'
        mig_profiles: List of available MIG profile dicts
        workloads: List of workload requirement dicts
    
    Returns:
        Dict with allocation results and utilization metrics
    """
    total_compute_slices = gpu_config["total_compute_slices"]
    total_memory_gb = gpu_config["total_memory_gb"]
    def find_best_mig(workload):
        compatible = [x for x in mig_profiles if x["compute_slices"] >= workload["min_compute_slices"] and \
                x["memory_gb"] >= workload["min_memory_gb"]]
        if not compatible:
            return None
        best = min(compatible, key=lambda x: (x["compute_slices"], x["memory_gb"]))
        return {
                "workload": workload["name"],
                "profile": best["name"],
                "compute_slices": best["compute_slices"],
                "memory_gb": best["memory_gb"],
                }


    workloads.sort(key=lambda x: (-x["min_compute_slices"], -x["min_memory_gb"]))
    workloads_served = 0
    workloads_rejected = [] 
    allocations = []
    remaining_compute_slices = total_compute_slices
    remaining_memory_gb = total_memory_gb
    for workload in workloads:
        allocation = find_best_mig(workload)
        if allocation and allocation["compute_slices"] <= remaining_compute_slices and allocation["memory_gb"] <= remaining_memory_gb:
            allocations.append(allocation)
            workloads_served += 1
            remaining_compute_slices -= allocation["compute_slices"]
            remaining_memory_gb -= allocation["memory_gb"]
        else:
            workloads_rejected.append(workload["name"])

    total_compute_used = sum(x["compute_slices"] for x in allocations)
    total_memory_used = sum(x["memory_gb"] for x in allocations)
    compute_utilization = total_compute_used / total_compute_slices * 100
    memory_utilization = total_memory_used / total_memory_gb * 100
    _ = lambda x: round(x, 2)
    return {
            "allocations": allocations,
            "total_compute_used": total_compute_used,
            "total_memory_used": total_memory_used,
            "compute_utilization": _(compute_utilization),
            "memory_utilization": _(memory_utilization),
            "workloads_served": workloads_served,
            "workloads_rejected": workloads_rejected,
            }