import itertools
from collections import defaultdict

def apriori(transactions, min_support=0.5, max_length=None):
    """
    Returns: dict mapping frozenset(itemset) -> support (float)
    """
    # TODO: Implement the Apriori algorithm
    if not transactions:
        raise ValueError
    items = set()
    for tr in transactions:
        items |= tr
    def support(t):
        ov = sum(t.issubset(tr) for tr in transactions)
        return ov / len(transactions)
    items = {frozenset([t]) for t in items if support(frozenset([t])) >= min_support}
    ans = {}
    xt = {frozenset()}
    def extend(ans, xt, items, l):
        new_xt = set() 
        for x in xt:
            for t in items:
                xe = x | t
                if len(xe) != l:
                    continue
                if xe in new_xt:
                    continue
                new_xt.add(xe)
                s = support(xe)
                if s >= min_support:
                    ans[xe] = s
        return ans