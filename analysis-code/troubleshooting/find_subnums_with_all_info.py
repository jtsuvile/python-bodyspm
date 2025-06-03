import numpy as np
import pandas as pd

subs_lbp_file = '/Volumes/Shield1/kipupotilaat/data/stockholm/stockholm_subnums_lbp.txt'
subs_fibro_file = '/Volumes/Shield1/kipupotilaat/data/stockholm/stockholm_subnums_fibromyalgi.txt'
subs_system_file = '/Volumes/Shield1/kipupotilaat/data/stockholm/stockholm_subnums_system_all.txt'


lbp = list(pd.read_csv(subs_lbp_file, header=None)[0])
fibro = list(pd.read_csv(subs_fibro_file, header=None)[0])
system = list(pd.read_csv(subs_system_file, header=None)[0])

fibro_out = list(set(fibro) & set(system))
lbp_out = list(set(lbp) & set(system))

str_out = f"""
patients in system {len(system)} 

patients in fibro list from doctors {len(fibro)}
patients in system and fibro list {len(fibro_out)}

patients in LBP list from doctors {len(lbp)}
patients in system and LBP list {len(lbp_out)}
"""
print(str_out)

with open('/Volumes/Shield1/kipupotilaat/data/stockholm/stockholm_subnums_fibro_pre_qc.txt', 'w') as f:
    for line in fibro_out:
        f.write(f"{line}\n")


with open('/Volumes/Shield1/kipupotilaat/data/stockholm/stockholm_subnums_lbp_pre_qc.txt', 'w') as f:
    for line in lbp_out:
        f.write(f"{line}\n")

with open('/Volumes/Shield1/kipupotilaat/data/stockholm/stockholm_combine_subnums_from_diff_sources.txt', 'w') as f:
    f.write(str_out)