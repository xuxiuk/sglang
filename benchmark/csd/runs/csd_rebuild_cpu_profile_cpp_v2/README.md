# CSD rebuild C++ CPU profile

Open `csd_rebuild_gil.chrometrace.json` in <https://ui.perfetto.dev/>.

This run loads the standalone test build of `csd_build_hash_table_cpu` and
executes 1000 production rebuilds. The Python `_try_build_hash_table` and
`_hash64` frames disappear because the native Torch operator executes without
holding the Python GIL.

| range | inclusive total | per rebuild |
|---|---:|---:|
| `rebuild_once` | 5318.6 ms | 5.32 ms |
| `merge_counts` | 1296.1 ms | 1.30 ms |
| `filtered_keys` | 582.0 ms | 0.58 ms |
| `build_csd_hash_table_payload` | 3364.1 ms | 3.36 ms |
| `_try_build_hash_table` | 0 ms | 0 ms |
| `_hash64` | 0 ms | 0 ms |

The native hash build itself takes about 0.26 ms for 34,305 keys, compared
with about 28.4 ms for the Python implementation. The remaining Python time
comes from `merge_counts`, membership-list materialization, and Tensor input
construction.
