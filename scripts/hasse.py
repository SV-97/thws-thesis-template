def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

def order_partition(p): 
    for start in p: 
        print(f'"{start}"')
        for end in p:
            if start == end: continue
            if max(start) <= min(end): 
                print(f'"{start}"->"{end}";')

parts = list(partition([1,2,3,4]))
for p in parts:
    order_partition(p)
