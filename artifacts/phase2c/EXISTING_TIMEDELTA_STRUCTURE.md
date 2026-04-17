# EXISTING_TIMEDELTA_STRUCTURE.md

## 1. Overview

`pd.Timedelta` represents a duration - the difference between two timestamps. It is the pandas equivalent of Python's `datetime.timedelta` but with nanosecond precision and rich parsing.

Key pandas objects:
- `pd.Timedelta` - scalar duration value
- `pd.TimedeltaIndex` - index of durations
- `pd.to_timedelta()` - flexible duration parser

## 2. Timedelta Constructor

### 2.1 Signature

```python
pd.Timedelta(
    value=<no_default>,
    unit=None,    # 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'
    **kwargs      # days, hours, minutes, seconds, milliseconds, microseconds, nanoseconds
)
```

### 2.2 Construction Patterns

```python
# From string (ISO 8601 duration or pandas shorthand)
pd.Timedelta("1 day")
pd.Timedelta("2 days 3 hours")
pd.Timedelta("1d 2h 30m")
pd.Timedelta("P1DT2H30M")  # ISO 8601
pd.Timedelta("00:30:00")   # HH:MM:SS

# From numeric + unit
pd.Timedelta(1, unit='D')      # 1 day
pd.Timedelta(3600, unit='s')   # 1 hour
pd.Timedelta(1500, unit='ms')  # 1.5 seconds

# From keyword arguments
pd.Timedelta(days=1, hours=2, minutes=30)
pd.Timedelta(weeks=1)

# From datetime.timedelta
pd.Timedelta(datetime.timedelta(days=1, hours=2))

# From numpy.timedelta64
pd.Timedelta(np.timedelta64(1, 'D'))

# Special values
pd.Timedelta('NaT')  # Not-a-Time (null duration)
pd.NaT               # Also works
```

### 2.3 String Parsing Grammar

Accepted formats:
- ISO 8601 duration: `P[n]Y[n]M[n]DT[n]H[n]M[n]S` (pandas only uses D/H/M/S)
- Shorthand: `<number><unit>` where unit is D/d, H/h, M/min, S/s, ms, us, ns
- Compound: `1 day 2 hours 30 minutes` or `1d2h30m`
- Time format: `HH:MM:SS.fraction`
- Negative: `-1 day` or `-P1D`

Unit aliases:
| Unit | Aliases |
|------|---------|
| weeks | W, w, week, weeks |
| days | D, d, day, days |
| hours | H, h, hour, hours, hr |
| minutes | M, m, min, minute, minutes, T |
| seconds | S, s, sec, second, seconds |
| milliseconds | ms, milli, millis, millisecond, milliseconds, L |
| microseconds | us, µs, micro, micros, microsecond, microseconds, U |
| nanoseconds | ns, nano, nanos, nanosecond, nanoseconds, N |

## 3. Properties and Attributes

### 3.1 Component Properties

```python
td = pd.Timedelta("2 days 3:04:05.006007008")

td.days              # 2 (int) - whole days
td.seconds           # 11045 (int) - seconds in current day (0-86399)
td.microseconds      # 6007 (int) - microseconds in current second (0-999999)
td.nanoseconds       # 8 (int) - nanoseconds in current microsecond (0-999)
```

### 3.2 Total Duration Accessors

```python
td.total_seconds()   # 183845.006007008 (float)

# .components NamedTuple
td.components
# Components(days=2, hours=3, minutes=4, seconds=5, milliseconds=6, microseconds=7, nanoseconds=8)
```

### 3.3 Resolution and Precision

```python
td.resolution_string  # 'D', 'H', 'T', 'S', 'L', 'U', 'N' based on lowest non-zero component
td.value              # int64 nanoseconds (internal representation)
td.asm8               # numpy.timedelta64 (for interop)
```

## 4. Arithmetic Operations

### 4.1 Timedelta + Timedelta

```python
pd.Timedelta("1d") + pd.Timedelta("2h")  # Timedelta('1 days 02:00:00')
pd.Timedelta("1d") - pd.Timedelta("2h")  # Timedelta('0 days 22:00:00')
```

### 4.2 Timedelta with Scalar

```python
pd.Timedelta("1d") * 3        # Timedelta('3 days')
pd.Timedelta("1d") / 2        # Timedelta('0 days 12:00:00')
pd.Timedelta("1d") // 2       # Timedelta('0 days 12:00:00') - floor division
pd.Timedelta("5d") % pd.Timedelta("2d")  # Timedelta('1 days')
```

### 4.3 Timedelta / Timedelta (Ratio)

```python
pd.Timedelta("2d") / pd.Timedelta("1d")  # 2.0 (float)
```

### 4.4 Timestamp Arithmetic

```python
ts = pd.Timestamp("2024-01-01")
ts + pd.Timedelta("1d")       # Timestamp('2024-01-02')
ts - pd.Timedelta("1d")       # Timestamp('2023-12-31')

ts2 = pd.Timestamp("2024-01-05")
ts2 - ts                       # Timedelta('4 days')
```

### 4.5 Series/DataFrame Arithmetic

```python
s = pd.Series([pd.Timedelta("1d"), pd.Timedelta("2d")])
s + pd.Timedelta("1h")        # Element-wise addition
s.sum()                        # Timedelta('3 days')
s.mean()                       # Timedelta('1 days 12:00:00')
```

### 4.6 Negation and Absolute

```python
-pd.Timedelta("1d")           # Timedelta('-1 days')
abs(pd.Timedelta("-1d"))      # Timedelta('1 days')
```

## 5. Comparison Operations

```python
pd.Timedelta("1d") > pd.Timedelta("12h")   # True
pd.Timedelta("1d") == pd.Timedelta("24h")  # True
pd.Timedelta("1d") >= pd.Timedelta("1d")   # True

# NaT comparisons
pd.Timedelta("1d") > pd.NaT                # False (NaT poisons)
pd.NaT == pd.NaT                           # False
pd.NaT != pd.NaT                           # True
```

## 6. Rounding and Floor/Ceil

```python
td = pd.Timedelta("1d 2h 35m 22s")

td.round('h')   # Timedelta('1 days 03:00:00')
td.floor('h')   # Timedelta('1 days 02:00:00')
td.ceil('h')    # Timedelta('1 days 03:00:00')
```

## 7. String Representations

```python
td = pd.Timedelta("1 days 02:30:45.123456")

str(td)                        # '1 days 02:30:45.123456000'
repr(td)                       # "Timedelta('1 days 02:30:45.123456000')"
td.isoformat()                 # 'P1DT2H30M45.123456S' (ISO 8601)
```

## 8. pd.to_timedelta() Function

### 8.1 Signature

```python
pd.to_timedelta(
    arg,              # scalar, list, array, Series, Index
    unit=None,        # unit for numeric arg
    errors='raise'    # 'raise', 'coerce', 'ignore'
)
```

### 8.2 Examples

```python
# Scalar
pd.to_timedelta("1 day")           # Timedelta('1 days')
pd.to_timedelta(1, unit='D')       # Timedelta('1 days')

# Series
pd.to_timedelta(pd.Series(["1d", "2h", "invalid"]), errors='coerce')
# 0   1 days 00:00:00
# 1   0 days 02:00:00
# 2               NaT

# Numeric with unit
pd.to_timedelta([1, 2, 3], unit='h')  # TimedeltaIndex

# Already Timedelta (passthrough)
pd.to_timedelta(pd.Timedelta("1d"))   # Timedelta('1 days')
```

### 8.3 Error Handling

```python
errors='raise'   # Raise ValueError for invalid input (default)
errors='coerce'  # Convert invalid to NaT
errors='ignore'  # Return input unchanged if conversion fails
```

## 9. TimedeltaIndex

### 9.1 Construction

```python
pd.TimedeltaIndex(["1d", "2d", "3d"])
pd.TimedeltaIndex([1, 2, 3], unit='h')
pd.timedelta_range(start="1 day", periods=5)
pd.timedelta_range(start="1 day", end="5 days", freq='D')
```

### 9.2 Properties

```python
tdi = pd.TimedeltaIndex(["1d", "2d", "3d"])

tdi.days           # Index([1, 2, 3], dtype='int64')
tdi.seconds        # Index([0, 0, 0], dtype='int64')
tdi.total_seconds() # Float64Index([86400.0, 172800.0, 259200.0])
tdi.components     # DataFrame with days, hours, minutes, etc.
```

### 9.3 Aggregations

```python
tdi.sum()    # Timedelta('6 days')
tdi.mean()   # Timedelta('2 days')
tdi.min()    # Timedelta('1 day')
tdi.max()    # Timedelta('3 days')
```

## 10. Internal Representation

- Stored as int64 nanoseconds internally
- Range: approximately ±292 years at nanosecond precision
- NaT represented as int64 minimum value (`np.iinfo(np.int64).min`)

### 10.1 Limits

```python
pd.Timedelta.min   # Timedelta('-106752 days +00:12:43.145224193')
pd.Timedelta.max   # Timedelta('106751 days 23:47:16.854775807')
pd.Timedelta.resolution  # Timedelta('0 days 00:00:00.000000001')
```

## 11. Null Handling

- `pd.NaT` (Not-a-Time) is the null value for Timedelta
- NaT propagates through arithmetic: `Timedelta + NaT = NaT`
- NaT comparisons return False (except `!=`)
- `pd.isna()` and `pd.isnull()` detect NaT

## 12. Type Coercion Rules

| From | To Timedelta |
|------|--------------|
| str | Parse as duration |
| int/float | Interpret with unit (default ns) |
| datetime.timedelta | Convert preserving semantics |
| numpy.timedelta64 | Convert preserving semantics |
| Timedelta | Passthrough |
| NaT/None/nan | NaT |

## 13. Edge Cases

### 13.1 Overflow

```python
# Exceeding ~292 years causes overflow
pd.Timedelta(days=107000)  # OverflowError
```

### 13.2 Precision Loss

```python
# Float seconds can lose nanosecond precision
pd.Timedelta(seconds=1.123456789123)  # May truncate beyond ns
```

### 13.3 Negative Timedeltas

```python
td = pd.Timedelta("-1d 2h")
td.days       # -1
td.seconds    # 7200 (positive - component within day)
```

## 14. Series Timedelta Accessor (.dt)

When a Series contains Timedelta values, `.dt` accessor provides:

```python
s = pd.Series(pd.to_timedelta(["1d", "2d 3h", "4d 5h 6m"]))

s.dt.days          # Series of day components
s.dt.seconds       # Series of second-of-day components
s.dt.total_seconds() # Series of total seconds
s.dt.components    # DataFrame of components
```

## 15. Priority Implementation Targets

### Phase 1: Core Type (MVP)
1. `Timedelta` scalar type with nanosecond storage
2. Construction from numeric + unit
3. Basic string parsing ("1 day", "2h", "1d 2h 30m")
4. Component properties (days, seconds, microseconds, nanoseconds)
5. `total_seconds()` method
6. Arithmetic: Timedelta ± Timedelta, Timedelta * scalar, Timedelta / scalar
7. Comparison operators
8. NaT handling

### Phase 2: Extended Support
1. Full ISO 8601 parsing
2. `pd.to_timedelta()` function with errors parameter
3. Timedelta ÷ Timedelta (ratio)
4. `round()`, `floor()`, `ceil()`
5. `isoformat()` output
6. Series Timedelta dtype integration

### Phase 3: Index Integration
1. TimedeltaIndex type
2. `timedelta_range()` constructor
3. Index aggregations (sum, mean, min, max)
4. DataFrame column Timedelta dtype
