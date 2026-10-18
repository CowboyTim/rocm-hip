#!/usr/bin/env python3
# prof_summary.py -- GPU time per kernel from a rocprofv3 --kernel-trace run (its SQLite results database).
#
#   rocprofv3 --kernel-trace -d <dir> -o run -- <command>
#   prof_summary.py <dir> [n]   -> total GPU time and launches, then the n (default 30) most expensive kernels

import glob, sqlite3, sys

d = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
db = glob.glob(d + "/**/*.db", recursive=True)
if not db:
    sys.exit(f"no results database under {d}")
c = sqlite3.connect(db[0])
rows = c.execute("select name, count(*), sum(end - start) / 1e6, avg(end - start) / 1e3 from kernels group by name "
                 "order by 3 desc").fetchall()
tot = sum(r[2] for r in rows)
n = c.execute("select count(*) from kernels").fetchone()[0]
print(f"total GPU {tot:.1f} ms, {n} launches")
for name, cnt, ms, avg in rows[:top]:
    print(f"{ms:9.1f} ms {100 * ms / tot:5.1f}% {cnt:7d} x {avg:8.1f} us  {name[:70]}")
