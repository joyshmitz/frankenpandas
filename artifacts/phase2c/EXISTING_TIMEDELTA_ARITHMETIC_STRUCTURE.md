# EXISTING_TIMEDELTA_ARITHMETIC_STRUCTURE.md

## 1. Overview

Timedelta Series support arithmetic operations for duration calculations. These operations work element-wise and maintain proper type semantics.

## 2. Timedelta + Timedelta

```python
td1 = pd.to_timedelta(['1d', '2d'])
td2 = pd.to_timedelta(['12h', '6h'])
td1 + td2  # ['1 days 12:00:00', '2 days 06:00:00']
td1 - td2  # ['0 days 12:00:00', '1 days 18:00:00']
```

Result type: Timedelta64

## 3. Timedelta + Scalar Timedelta

```python
td = pd.to_timedelta(['1d', '2d'])
td + pd.Timedelta('6h')  # ['1 days 06:00:00', '2 days 06:00:00']
td - pd.Timedelta('6h')  # ['0 days 18:00:00', '1 days 18:00:00']
```

Result type: Timedelta64

## 4. Timedelta * Numeric

```python
td = pd.to_timedelta(['1d', '2d'])
td * 2        # ['2 days', '4 days']
td * 0.5      # ['0 days 12:00:00', '1 days 00:00:00']
2 * td        # ['2 days', '4 days'] (commutative)
```

Result type: Timedelta64

## 5. Timedelta / Numeric

```python
td = pd.to_timedelta(['1d', '2d'])
td / 2        # ['0 days 12:00:00', '1 days 00:00:00']
td // 2       # ['0 days 12:00:00', '1 days 00:00:00'] (floor division)
```

Result type: Timedelta64

## 6. Timedelta / Timedelta

```python
td1 = pd.to_timedelta(['2d', '4d'])
td2 = pd.to_timedelta(['1d', '2d'])
td1 / td2     # [2.0, 2.0] - Float64 ratio
```

Result type: Float64 (duration ratio)

## 7. Timedelta % Timedelta

```python
td1 = pd.to_timedelta(['25h', '50h'])
td2 = pd.to_timedelta(['24h', '24h'])
td1 % td2     # ['0 days 01:00:00', '0 days 02:00:00']
```

Result type: Timedelta64

## 8. Unary Operations

```python
td = pd.to_timedelta(['1d', '-2d'])
-td           # ['-1 days', '2 days']
abs(td)       # ['1 days', '2 days']
```

## 9. Comparison Operations

```python
td1 = pd.to_timedelta(['1d', '2d'])
td2 = pd.to_timedelta(['2d', '1d'])

td1 < td2     # [True, False]
td1 > td2     # [False, True]
td1 == td2    # [False, False]
td1 != td2    # [True, True]
td1 <= td2    # [True, False]
td1 >= td2    # [False, True]
```

Result type: Bool

## 10. NaT Handling

```python
td1 = pd.to_timedelta(['1d', pd.NaT])
td2 = pd.to_timedelta(['1d', '1d'])

td1 + td2     # ['2 days', NaT]
td1 * 2       # ['2 days', NaT]
td1 / 2       # ['12:00:00', NaT]
td1 == td2    # [True, False] - NaT != anything
td1 < td2     # [False, False] - comparisons with NaT are False
```

## 11. Type Coercion

| Operation | Left Type | Right Type | Result Type |
|-----------|-----------|------------|-------------|
| + | Timedelta | Timedelta | Timedelta |
| - | Timedelta | Timedelta | Timedelta |
| * | Timedelta | Int/Float | Timedelta |
| * | Int/Float | Timedelta | Timedelta |
| / | Timedelta | Int/Float | Timedelta |
| / | Timedelta | Timedelta | Float64 |
| // | Timedelta | Int/Float | Timedelta |
| % | Timedelta | Timedelta | Timedelta |

## 12. Priority Implementation Targets

### Phase 1: Core Arithmetic
1. `Series::td_add(other)` - timedelta + timedelta
2. `Series::td_sub(other)` - timedelta - timedelta
3. `Series::td_mul(scalar)` - timedelta * numeric
4. `Series::td_div(scalar)` - timedelta / numeric

### Phase 2: Extended Operations
1. `Series::td_ratio(other)` - timedelta / timedelta → float
2. `Series::td_mod(other)` - timedelta % timedelta
3. `Series::td_floordiv(scalar)` - timedelta // numeric
4. `Series::td_neg()` - unary negation
5. `Series::td_abs()` - absolute value

### Phase 3: Comparison
1. `Series::td_lt(other)` - less than
2. `Series::td_le(other)` - less than or equal
3. `Series::td_gt(other)` - greater than
4. `Series::td_ge(other)` - greater than or equal
5. `Series::td_eq(other)` - equality
6. `Series::td_ne(other)` - inequality
