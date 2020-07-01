import pstats
from pstats import SortKey
p = pstats.Stats('disko.prof')
p.sort_stats(SortKey.TIME).print_stats(20)
