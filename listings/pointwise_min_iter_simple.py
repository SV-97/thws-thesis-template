from typing import List
from pointwise_min_rec import AffineFn, PcwAffFn, graph_intersection

def pointwise_min(affines: List[AffineFn]) -> PcwAffFn:
    """`affines` must be strictly sorted by slope in descending order"""
    match len(affines):
        case 0:
            return None # an empty set has no minimum
        case 1:
            return PcwAffFn(borders=[], fns=affines)
        case _:
            # stack for affine segments of minimum. We initially assume the
            # first two affine functions to be part of the minimum.
            minimals = [affines[0], affines[1]]
            # stack for locations where one affine segment switches over into
            # the next one. We initially assume that the first jump happens where
            # the first two functions intersect.
            jumps = [graph_intersection(affines[0], affines[1])]
            # scan through the remaining functions
            for right_fn in affines[2:]:
                left_fn = minimals[-1]
                # verify assumptions made up until now with the new function
                while len(jumps) != 0 \
                        and (right_fn - left_fn)(jumps[-1]) <= 0.0:
                    # if the current left_fn is larger at the last jump location
                    # we know that it's non optimal on the whole interval
                    # (jumps[-2], jumps[-1]) (where jumps[-2] = -inf if it doesn't
                    # exist). Since slope(right_fn) < slope(left_fn) it'll also be
                    # larger than right_fn on (jumps[-1], inf). As such it's not
                    # part of the minimum and can be removed.
                    minimals.pop()
                    jumps.pop()
                    left_fn = minimals[-1]
                # assume a new right function to be part of the minimum
                jumps.append(graph_intersection(left_fn, right_fn))
                minimals.append(right_fn)
            return PcwAffFn(borders=jumps, fns=minimals)
