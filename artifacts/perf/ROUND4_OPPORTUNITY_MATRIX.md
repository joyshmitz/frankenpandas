# ROUND4 Opportunity Matrix

| Hotspot | Impact | Confidence | Effort | Score |
|---|---:|---:|---:|---:|
| Hash-heavy group-key accumulation in `groupby_sum` | 3 | 4 | 4 | 3.00 |
| Duplicate-check hash path in `Index::has_duplicates` | 5 | 5 | 2 | 12.50 |
| Repeated full index equality in fast-path guard | 4 | 4 | 3 | 5.33 |

Round-4 selected lever:
- Dense `Int64` bucket path for `groupby_sum` with bounded-range fallback.

Reason:
- Score `>= 2.0`, low semantic risk due explicit fallback to generic path.
