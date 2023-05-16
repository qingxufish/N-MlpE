import pstats

stats = pstats.Stats('result.out')
stats.strip_dirs().sort_stats('tottime').print_stats()