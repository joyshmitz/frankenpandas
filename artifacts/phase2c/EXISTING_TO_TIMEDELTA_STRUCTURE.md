# EXISTING_TO_TIMEDELTA_STRUCTURE.md

## 1. Overview

`pd.to_timedelta()` converts various inputs to Timedelta or TimedeltaIndex. Used for parsing string durations, converting numeric values with units, and coercing timedelta-like objects.

Key function:
- `pd.to_timedelta(arg, unit=None, errors='raise')`

## 2. Function Signature

```python
pd.to_timedelta(
    arg,            # scalar, array, list, or Series
    unit=None,      # unit for numeric input: 'D','h','m','s','ms','us','ns'
    errors='raise', # 'raise', 'coerce', 'ignore'
)
```

## 3. Input Types and Return Types

| Input Type | Return Type |
|------------|-------------|
| str | Timedelta |
| int/float | Timedelta (with unit) |
| timedelta | Timedelta |
| np.timedelta64 | Timedelta |
| list/array | TimedeltaIndex |
| Series | Series (timedelta64[ns] dtype) |

## 4. String Parsing

### 4.1 ISO 8601 Duration (P-prefix)

```python
pd.to_timedelta('P1DT2H30M')  # 1 day, 2 hours, 30 minutes
pd.to_timedelta('PT1H')       # 1 hour
pd.to_timedelta('P1W')        # 1 week = 7 days
```

### 4.2 Pandas-Style Compound

```python
pd.to_timedelta('1 day 2 hours 30 minutes')
pd.to_timedelta('1d 2h 30m')
pd.to_timedelta('1 days 02:30:00')  # mixed format
```

### 4.3 Simple Units

```python
pd.to_timedelta('5 days')
pd.to_timedelta('3h')
pd.to_timedelta('45m')
pd.to_timedelta('100ms')
```

### 4.4 Time Format (HH:MM:SS)

```python
pd.to_timedelta('01:30:00')     # 1 hour 30 minutes
pd.to_timedelta('01:30:00.500') # with milliseconds
pd.to_timedelta('-01:30:00')    # negative duration
```

### 4.5 Negative Durations

```python
pd.to_timedelta('-1 day')
pd.to_timedelta('-2h 30m')
```

## 5. Numeric Input with Unit

### 5.1 Unit Aliases

| Unit | Aliases | Meaning |
|------|---------|---------|
| D | 'D', 'day', 'days' | days |
| h | 'h', 'H', 'hour', 'hours', 'hr' | hours |
| m | 'm', 'min', 'minute', 'minutes', 'T' | minutes |
| s | 's', 'S', 'sec', 'second', 'seconds' | seconds |
| ms | 'ms', 'L', 'milli', 'millis', 'millisecond', 'milliseconds' | milliseconds |
| us | 'us', 'U', 'micro', 'micros', 'microsecond', 'microseconds' | microseconds |
| ns | 'ns', 'N', 'nano', 'nanos', 'nanosecond', 'nanoseconds' | nanoseconds |
| W | 'W', 'week', 'weeks' | weeks |

### 5.2 Examples

```python
pd.to_timedelta(5, unit='D')   # Timedelta('5 days')
pd.to_timedelta(1.5, unit='h') # Timedelta('0 days 01:30:00')
pd.to_timedelta([1, 2, 3], unit='D')  # TimedeltaIndex
```

## 6. Error Handling

### 6.1 errors='raise' (default)

```python
pd.to_timedelta('invalid')  # raises ValueError
```

### 6.2 errors='coerce'

```python
pd.to_timedelta('invalid', errors='coerce')  # NaT
pd.to_timedelta(['1d', 'invalid'], errors='coerce')  # [Timedelta('1 days'), NaT]
```

### 6.3 errors='ignore'

```python
pd.to_timedelta('invalid', errors='ignore')  # returns 'invalid' unchanged
```

## 7. Series Input

```python
s = pd.Series(['1d', '2h', '30m'])
pd.to_timedelta(s)  # Series with timedelta64[ns] dtype
```

## 8. NaN/NaT Handling

```python
pd.to_timedelta(np.nan)       # NaT
pd.to_timedelta([1, np.nan], unit='D')  # [Timedelta('1 days'), NaT]
pd.to_timedelta('NaT')        # NaT
```

## 9. Edge Cases

### 9.1 Empty Input

```python
pd.to_timedelta([])  # TimedeltaIndex([], dtype='timedelta64[ns]')
```

### 9.2 Overflow

```python
pd.to_timedelta(1e20, unit='D')  # raises OverflowError (too large)
```

### 9.3 Precision

```python
pd.to_timedelta(0.001, unit='ns')  # rounds to 0 ns
```

## 10. Internal Representation

- All values stored as int64 nanoseconds
- NaT = int64::MIN (same as Timedelta scalar)
- Maximum representable: ~292 years
- Minimum representable: ~-292 years

## 11. Priority Implementation Targets

### Phase 1: Core `to_timedelta()`
1. Module-level `to_timedelta(arg, unit, errors)` function
2. String parsing (reuse `Timedelta::parse`)
3. Numeric conversion with unit multiplier
4. Scalar return for single values

### Phase 2: Collection Support
1. Vec<Scalar>/slice input → TimedeltaIndex
2. Series input → Series with timedelta64[ns] dtype
3. NaN/NaT coercion

### Phase 3: Error Handling
1. `errors='raise'` default
2. `errors='coerce'` → NaT
3. `errors='ignore'` → pass-through
