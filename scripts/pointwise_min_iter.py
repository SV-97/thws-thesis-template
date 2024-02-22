from fractions import Fraction
from numbers import Real
from typing import List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np


LINEWIDTH = 4


class AffineFn(NamedTuple):
    slope: Real
    intercept: Real

    def __call__(self, x):
        return self.slope * x + self.intercept

    def __sub__(self, other):
        return AffineFn(self.slope - other.slope, self.intercept - other.intercept)


def graph_intersection(fn1: AffineFn, fn2: AffineFn) -> Real:
    return (fn1.intercept - fn2.intercept) / (fn2.slope - fn1.slope)


class PcwAffFn(NamedTuple):
    borders: List[Real]
    fns: List[AffineFn]


def pointwise_min(affines: List[AffineFn]) -> PcwAffFn:
    """`affines` must be strictly sorted by slope in descending order"""
    match len(affines):
        case 0:
            return None # an empty set has no minimum
        case 1:
            return PcwAffFn(borders=[], fns=affines)
        case n:
            # stack for affine segments of minimum
            # we initially assume the first two affine functions
            # to be part of the minmium
            minimals = [affines[0], affines[1]]
            # stack for locations where one affine segment switches over
            # into the next one.
            # We initially assume that the first jump happens where
            # the first two functions intersect
            jumps = [graph_intersection(affines[0], affines[1])]
            # scan through the remaining functions
            for right_fn in affines[2:]:
                left_fn = minimals[-1]
                # verify assumptions made up until now with the new function
                while len(jumps) != 0 \
                        and (right_fn - left_fn)(jumps[-1]) <= 0.0:
                    # if the current left_fn is larger at the last jump location
                    # we know that it's non optimal on the whole interval (jumps[-2], jumps[-1])
                    # (where jumps[-2] = -inf if it doesn't exist). But since
                    #   slope(right_fn) < slope(left_fn) it'll also be larger than right_fn
                    # on (jumps[-1], inf). As such it's not part of the minimum and can be removed.
                    minimals.pop()
                    jumps.pop()
                    left_fn = minimals[-1]
                # we now assume that the new jump is optimal
                new_jump = graph_intersection(left_fn, right_fn)
                # for highly degenerate inputs it may be the case that the newly added jump
                # is smaller than it should be because of numerical problems. To provide
                # a usually very good approximation of the correct solution in those cases
                # we scan through the previous jumps removing the ones that are too large
                while len(jumps) != 0 and new_jump <= jumps[-1]:
                    # a newly added jump shouldn't be smaller than a previous jump in any case
                    jumps.pop()
                    minimals.pop()
                # once everything is okay we just add the new jump and function segment
                jumps.append(new_jump)
                minimals.append(right_fn)
            return PcwAffFn(borders=jumps, fns=minimals)


#affines = [AffineFn(Fraction(i, 1), Fraction(1, i)) for i in range(1, 21, 2)]
affines = affines = [AffineFn(5, 0), AffineFn(3, 2), AffineFn(1, 4), AffineFn(0, 6), AffineFn(-2, 20), AffineFn(0.25, 2)]
solution = pointwise_min(sorted(affines, key=lambda fn: fn.slope, reverse=True))
# print(solution)
affines = sorted(affines, key=lambda fn: fn.slope, reverse=True)


def sample_pcw_fn(pcw_aff: PcwAffFn, x_min: Real, x_max: Real):
    if len(pcw_aff.borders) == 0:
        boundaries = [x_min, x_max]
    else:
        boundaries = [min(min(pcw_aff.borders) - 1, x_min)]
        boundaries.extend(pcw_aff.borders)
        boundaries.append(max(max(pcw_aff.borders) + 1, x_max))
    xs = []
    ys = []
    for i, fn_i in enumerate(pcw_aff.fns):
        xs_i = np.linspace(boundaries[i], boundaries[i+1])
        ys_i = fn_i(xs_i)
        xs.append(xs_i)
        ys.append(ys_i)
    return np.hstack(xs), np.hstack(ys)



xs_pcw, _ = sample_pcw_fn(solution, -1, 20)
fig, axs = plt.subplots(2, 3, sharey=True, sharex=True)
axs = [ax for axr in axs for ax in axr]

def plot_everything(ax, affines, **kwargs):
    if len(affines) == 0:
        return
    xs = xs_pcw
    ys = []
    for aff in affines:
        ys.append(aff(xs))
    if len(affines) > 1:
        kw = {key: value for key, value in kwargs.items() if key != "label"}
        ax.plot(xs.reshape(-1, 1), np.array(ys[1:]).T, **kw)
    ax.plot(xs.reshape(-1, 1), np.array(ys[0]).T, **kwargs)


match len(affines):
    case 0:
        solution = None # an empty set has no minimum
    case 1:
        solution = PcwAffFn(borders=[], fns=affines)
    case n:
        # stack for affine segments of minimum
        # we initially assume the first two affine functions
        # to be part of the minimum
        minimals = [affines[0], affines[1]]
        # stack for locations where one affine segment switches over
        # into the next one.
        # We initially assume that the first jump happens where
        # the first two functions intersect
        jumps = [graph_intersection(affines[0], affines[1])]
        
        ax = axs[0]
        # functions not considered yet
        plot_everything(ax, [aff for aff in affines if aff not in minimals], linestyle="--", alpha=0.5, color="tab:grey", label="Not considered yet", linewidth=LINEWIDTH)
        # current assumed minimum
        ax.plot(*sample_pcw_fn(PcwAffFn(jumps, minimals), xs_pcw[0], xs_pcw[-1]), label="Assumed $F$", linewidth=LINEWIDTH)
        plot_everything(ax, minimals, linestyle=":", alpha=0.5, color="tab:grey", linewidth=LINEWIDTH)
        # current right function
        # ax.plot(*sample_pcw_fn(PcwAffFn([], [affines[1]]), xs_pcw[0], xs_pcw[-1]), "-.", color="tab:orange", label="$f_r$")

        # scan through the remaining functions
        for i, right_fn in enumerate(affines[2:]):
            left_fn = minimals[-1]
            # verify assumptions made up until now with the new function
            while len(jumps) != 0 \
                    and (right_fn - left_fn)(jumps[-1]) <= 0.0:
                # if the current left_fn is larger at the last jump location
                # we know that it's non optimal on the whole interval (jumps[-2], jumps[-1])
                # (where jumps[-2] = -inf if it doesn't exist). But since
                #   slope(right_fn) < slope(left_fn) it'll also be larger than right_fn
                # on (jumps[-1], inf). As such it's not part of the minimum and can be removed.
                minimals.pop()
                jumps.pop()
                left_fn = minimals[-1]
            # we now assume that the new jump is optimal
            new_jump = graph_intersection(left_fn, right_fn)
            # for highly degenerate inputs it may be the case that the newly added jump
            # is smaller than it should be because of numerical problems. To provide
            # a usually very good approximation of the correct solution in those cases
            # we scan through the previous jumps removing the ones that are too large
            while len(jumps) != 0 and new_jump <= jumps[-1]:
                # a newly added jump shouldn't be smaller than a previous jump in any case
                jumps.pop()
                minimals.pop()
            # once everything is okay we just add the new jump and function segment
            jumps.append(new_jump)
            minimals.append(right_fn)
            
            ax = axs[i+1]
            # functions not considered yet
            plot_everything(ax, affines[3+i:], linestyle="--", alpha=0.5, color="tab:grey", label="Not considered yet", linewidth=LINEWIDTH)
            # all functions except the nonminimal ones
            nonminimal = [aff for aff in affines[:2+i] if aff not in minimals]
            plot_everything(ax, [aff for aff in affines if aff not in nonminimal], linestyle=":", alpha=0.5, color="tab:grey", linewidth=LINEWIDTH)
            # functions that we know to be nonminimal
            plot_everything(ax, nonminimal, alpha=0.8, color="tab:red", label="Known to be nonminimal", linewidth=LINEWIDTH)
            # current assumed minimum
            ax.plot(*sample_pcw_fn(PcwAffFn(jumps, minimals), xs_pcw[0], xs_pcw[-1]), label="Assumed $F$", linewidth=LINEWIDTH)
            # current right function
            ax.plot(*sample_pcw_fn(PcwAffFn([], [right_fn]), xs_pcw[0], xs_pcw[-1]), "-.", color="tab:orange", alpha=1, label="Current $f_r$", linewidth=LINEWIDTH)
        solution = PcwAffFn(borders=jumps, fns=minimals)

# plt.legend()
ax = axs[-1]
# functions not considered yet
# all functions except the nonminimal ones
nonminimal = [aff for aff in affines if aff not in minimals]
plot_everything(ax, [aff for aff in affines if aff not in nonminimal], linestyle=":", alpha=0.5, color="tab:grey", linewidth=LINEWIDTH)
# functions that we know to be nonminimal
plot_everything(ax, nonminimal, alpha=0.8, color="tab:red", label="Known to be nonminimal", linewidth=LINEWIDTH)
# current assumed minimum
ax.plot(*sample_pcw_fn(PcwAffFn(jumps, minimals), xs_pcw[0], xs_pcw[-1]), label="Assumed $F$", linewidth=LINEWIDTH)

fig.subplots_adjust(
    top=0.985,
    bottom=0.04,
    left=0.02,
    right=0.985,
    hspace=0.075,
    wspace=0.075
)


# xs_pcw, ys_pcw = sample_pcw_fn(solution, 0, 1)
# xs = xs_pcw
# ys = []
# for aff in affines:
#     ys.append(aff(xs))
# plt.plot(xs.reshape(-1, 1), np.array(ys).T, color="tab:blue")
# plt.plot(xs_pcw, ys_pcw, color="tab:orange")
plt.show()
