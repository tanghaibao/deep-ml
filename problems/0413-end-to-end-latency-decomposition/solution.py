import numpy as np

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    """
    Decompose end-to-end inference latency into component stages.
    
    Args:
        stage_latencies: dict mapping stage name -> np.ndarray of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])
    
    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """
    def pp(x, percentiles):
        return dict(zip(percentiles, np.percentile(x, percentiles).round(2)))

    _ = lambda x: round(x, 2)

    def get_stats(x):
        return {
                "mean": _(x.mean()),
                "std": _(x.std()),
                "percentiles": pp(x, percentiles),
                }
    stages = list(stage_latencies.keys())     
    S = np.row_stack([stage_latencies[x] for x in stages])
    L = S.sum(axis=0)
    e2e_mean = L.mean()
    e2e_percentile