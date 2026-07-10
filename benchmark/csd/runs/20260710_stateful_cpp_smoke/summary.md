
## apps_2000_code_ood
| method | tok/s | head90 tok/s | win30 p90 tok/s | win30 p90 tok | win30 max tok | win60 p90 tok/s | Δtok/s vs plain | accept | spec% | avg out | p90 out | max out | hit | hit% | tail tok% | tail time% | avg lat | p90 lat | p90 req tok/s | avg prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eagle` | 415.3 | 316.8 | 415.4 | 1024 | 1024 | 415.4 | - | 3.471 | 49.42 | 256.0 | 256 | 256 | 4/4 | 100.00% | 25.0% | 0.0% | 2.3 | 2.5 | 121.8 | 328.0 |
| `dynamic_ignore_ratio` | 911.1 | 689.3 | 911.3 | 1024 | 1024 | 911.3 | - | 4.096 | 61.92 | 256.0 | 256 | 256 | 4/4 | 100.00% | 25.0% | 0.0% | 1.1 | 1.1 | 260.8 | 328.0 |
| `dynamic_entropy_p20_ignore_ratio` | 816.4 | 678.2 | 816.5 | 1024 | 1024 | 816.5 | - | 3.793 | 55.86 | 256.0 | 256 | 256 | 4/4 | 100.00% | 25.0% | 0.0% | 1.1 | 1.3 | 251.1 | 328.0 |

## Candidate positives
- None passed the configured positive threshold.
