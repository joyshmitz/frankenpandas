# EXISTING_TIMEDELTAINDEX_STRUCTURE.md

## 1. Overview

`pd.TimedeltaIndex` is an Index of Timedelta values. Used for duration-based indexing and time series operations involving durations.

Key pandas objects:
- `pd.TimedeltaIndex` - index of duration values
- `pd.timedelta_range()` - range constructor for TimedeltaIndex

## 2. TimedeltaIndex Constructor

### 2.1 Signature

```python
pd.TimedeltaIndex(
    data=None,           # array-like of timedelta-like
    unit=None,           # unit for numeric data
    freq=None,           # offset alias or DateOffset
    closed=None,         # 'left', 'right', or None for intervals
    dtype=None,          # always timedelta64[ns]
    copy=False,
    name=None,
)
```

### 2.2 Construction Patterns

```python
# From strings
pd.TimedeltaIndex(['1 day', '2 days', '3 days'])

# From Timedelta objects
pd.TimedeltaIndex([pd.Timedelta('1d'), pd.Timedelta('2d')])

# From numeric + unit
pd.TimedeltaIndex([1, 2, 3], unit='D')

# From numpy timedelta64
pd.TimedeltaIndex(np.array([1, 2, 3], dtype='timedelta64[D]'))

# With name
pd.TimedeltaIndex(['1h', '2h'], name='duration')
```

## 3. timedelta_range() Function

### 3.1 Signature

```python
pd.timedelta_range(
    start=None,       # Timedelta or str
    end=None,         # Timedelta or str
    periods=None,     # int, number of periods
    freq=None,        # str or DateOffset, default 'D'
    name=None,        # name for resulting index
    closed=None,      # 'left', 'right', or None
)
```

### 3.2 Examples

```python
# By start and periods
pd.timedelta_range(start='1 day', periods=5)
# TimedeltaIndex(['1 days', '2 days', '3 days', '4 days', '5 days'])

# By start and end
pd.timedelta_range(start='1 day', end='5 days')
# TimedeltaIndex(['1 days', '2 days', '3 days', '4 days', '5 days'])

# With frequency
pd.timedelta_range(start='1 day', periods=5, freq='6h')
# TimedeltaIndex(['1 days 00:00:00', '1 days 06:00:00', ...])

# Negative range
pd.timedelta_range(start='-1 day', periods=3, freq='12h')

# With name
pd.timedelta_range(start='0', periods=3, freq='h', name='hours')
```

### 3.3 Frequency Aliases

| Alias | Meaning |
|-------|---------|
| D | day |
| h, H | hour |
| min, T | minute |
| s, S | second |
| ms, L | millisecond |
| us, U | microsecond |
| ns, N | nanosecond |

Numeric prefix allowed: '2D', '6h', '30min', etc.

## 4. Properties

### 4.1 Component Properties

```python
tdi = pd.timedelta_range('1 day', periods=3)

tdi.days           # Int64Index([1, 2, 3])
tdi.seconds        # Int64Index([0, 0, 0])
tdi.microseconds   # Int64Index([0, 0, 0])
tdi.nanoseconds    # Int64Index([0, 0, 0])
tdi.components     # DataFrame of all components
```

### 4.2 Duration Properties

```python
tdi.total_seconds()  # Float64Index of total seconds
```

### 4.3 Index Properties

```python
tdi.dtype        # dtype('timedelta64[ns]')
tdi.freq         # frequency if regular, else None
tdi.inferred_freq # inferred frequency
tdi.name         # index name
```

## 5. Aggregation Methods

```python
tdi = pd.TimedeltaIndex(['1d', '2d', '3d'])

tdi.sum()    # Timedelta('6 days')
tdi.mean()   # Timedelta('2 days')
tdi.min()    # Timedelta('1 day')
tdi.max()    # Timedelta('3 days')
tdi.std()    # Timedelta standard deviation
```

## 6. Arithmetic Operations

### 6.1 TimedeltaIndex + TimedeltaIndex

```python
tdi1 = pd.timedelta_range('1d', periods=3)
tdi2 = pd.timedelta_range('1h', periods=3, freq='h')
tdi1 + tdi2  # Element-wise addition
```

### 6.2 TimedeltaIndex + Timedelta

```python
tdi + pd.Timedelta('1h')  # Broadcast addition
```

### 6.3 TimedeltaIndex * Scalar

```python
tdi * 2        # Double all durations
tdi / 2        # Halve all durations
tdi // 2       # Floor divide
```

### 6.4 TimedeltaIndex / TimedeltaIndex

```python
tdi1 / tdi2    # Float64Index of ratios
```

## 7. Rounding Methods

```python
tdi = pd.timedelta_range('1d 2h 30m', periods=3, freq='h')

tdi.round('h')   # Round to nearest hour
tdi.floor('h')   # Floor to hour
tdi.ceil('h')    # Ceiling to hour
```

## 8. Slicing and Selection

```python
tdi = pd.timedelta_range('0', periods=10, freq='h')

tdi[0]           # Timedelta('0 days')
tdi[1:5]         # TimedeltaIndex slice
tdi[tdi > pd.Timedelta('5h')]  # Boolean mask
```

## 9. Set Operations

```python
tdi1 = pd.timedelta_range('1d', periods=5)
tdi2 = pd.timedelta_range('3d', periods=5)

tdi1.union(tdi2)        # Union
tdi1.intersection(tdi2) # Intersection
tdi1.difference(tdi2)   # Difference
```

## 10. Conversion Methods

```python
tdi.to_pytimedelta()   # array of datetime.timedelta
tdi.to_numpy()         # numpy timedelta64 array
tdi.to_series()        # Series with TimedeltaIndex
tdi.to_frame(name)     # DataFrame with single column
```

## 11. Internal Representation

- Stored as int64 nanoseconds (same as Timedelta scalar)
- NaT represented as int64 minimum value
- dtype is always `timedelta64[ns]`

## 12. Edge Cases

### 12.1 Empty Index

```python
pd.TimedeltaIndex([])  # Empty TimedeltaIndex with timedelta64[ns] dtype
```

### 12.2 NaT Values

```python
pd.TimedeltaIndex(['1d', pd.NaT, '3d'])  # Contains NaT
```

### 12.3 Frequency Inference

```python
# Regular spacing → freq is inferred
pd.TimedeltaIndex(['1d', '2d', '3d']).freq  # 'D'

# Irregular → freq is None
pd.TimedeltaIndex(['1d', '3d', '7d']).freq  # None
```

## 13. Priority Implementation Targets

### Phase 1: Core TimedeltaIndex
1. `TimedeltaIndex` struct wrapping existing `Index`
2. Construction from Vec<i64> nanoseconds
3. `timedelta_range()` with start/end/periods/freq
4. Component properties (days, seconds, total_seconds)
5. Basic aggregations (sum, mean, min, max)

### Phase 2: Operations
1. Arithmetic: add, sub, mul, div with scalars and indices
2. Rounding: round, floor, ceil
3. Set operations: union, intersection, difference

### Phase 3: Integration
1. Series with TimedeltaIndex
2. DataFrame indexing with TimedeltaIndex
3. Conformance packet fixtures
