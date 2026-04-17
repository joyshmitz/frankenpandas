# PROPOSED_TIMEDELTA_ARCHITECTURE.md

## 1. Design Goals

1. **Nanosecond precision** - Match pandas i64 nanosecond storage
2. **Clean integration** - Extend existing DType/Scalar/Column infrastructure
3. **Type safety** - Leverage Rust's type system to prevent invalid operations
4. **Minimal footprint** - Add only necessary complexity

## 2. Type System Extensions

### 2.1 DType Extension (fp-types)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    Null,
    Bool,
    Int64,
    Float64,
    Utf8,
    Timedelta64,  // NEW: nanosecond duration
}
```

### 2.2 Scalar Extension (fp-types)

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    Null(NullKind),
    Bool(bool),
    Int64(i64),
    Float64(f64),
    Utf8(String),
    Timedelta64(i64),  // NEW: nanoseconds, i64::MIN = NaT
}
```

### 2.3 Timedelta Helper Struct (fp-types)

```rust
/// Constants and helpers for Timedelta operations.
/// The actual storage remains i64 nanoseconds in Scalar::Timedelta64.
pub struct Timedelta;

impl Timedelta {
    pub const NANOS_PER_MICRO: i64 = 1_000;
    pub const NANOS_PER_MILLI: i64 = 1_000_000;
    pub const NANOS_PER_SEC: i64 = 1_000_000_000;
    pub const NANOS_PER_MIN: i64 = 60 * Self::NANOS_PER_SEC;
    pub const NANOS_PER_HOUR: i64 = 60 * Self::NANOS_PER_MIN;
    pub const NANOS_PER_DAY: i64 = 24 * Self::NANOS_PER_HOUR;
    pub const NANOS_PER_WEEK: i64 = 7 * Self::NANOS_PER_DAY;
    
    pub const NAT: i64 = i64::MIN;  // Not-a-Time sentinel
    
    /// Parse duration string to nanoseconds
    pub fn parse(s: &str) -> Result<i64, TimedeltaError>;
    
    /// Format nanoseconds as duration string  
    pub fn format(nanos: i64) -> String;
    
    /// Extract components from nanoseconds
    pub fn components(nanos: i64) -> TimedeltaComponents;
}

#[derive(Debug, Clone, Copy)]
pub struct TimedeltaComponents {
    pub days: i64,
    pub hours: i64,      // 0-23
    pub minutes: i64,    // 0-59
    pub seconds: i64,    // 0-59
    pub milliseconds: i64,  // 0-999
    pub microseconds: i64,  // 0-999
    pub nanoseconds: i64,   // 0-999
}
```

## 3. Parsing Strategy

### 3.1 Supported Formats (Phase 1)

1. **Simple numeric + unit**: `"1d"`, `"2h"`, `"30m"`, `"45s"`, `"100ms"`
2. **Compound**: `"1d 2h 30m"`, `"1 day 2 hours"`
3. **Time format**: `"01:30:00"` (HH:MM:SS)

### 3.2 Parser Implementation

```rust
/// Parse duration string to nanoseconds.
/// Returns i64::MIN for "NaT" or invalid input with errors='coerce'.
pub fn parse_timedelta(s: &str) -> Result<i64, TimedeltaError> {
    let s = s.trim();
    
    // Check for NaT
    if s.eq_ignore_ascii_case("nat") {
        return Ok(Timedelta::NAT);
    }
    
    // Try time format HH:MM:SS
    if let Some(nanos) = try_parse_time_format(s) {
        return Ok(nanos);
    }
    
    // Try compound format "1d 2h 30m"
    parse_compound_duration(s)
}

fn try_parse_time_format(s: &str) -> Option<i64> {
    // Match HH:MM:SS or HH:MM:SS.fraction
    let re = Regex::new(r"^(-?)(\d+):(\d{2}):(\d{2})(?:\.(\d+))?$").ok()?;
    // ... implementation
}

fn parse_compound_duration(s: &str) -> Result<i64, TimedeltaError> {
    // Match repeated <number><unit> patterns
    // Handle negative prefix
    // Sum all components
}
```

### 3.3 Unit Mapping

```rust
fn unit_to_nanos(unit: &str) -> Option<i64> {
    match unit.to_lowercase().as_str() {
        "w" | "week" | "weeks" => Some(Timedelta::NANOS_PER_WEEK),
        "d" | "day" | "days" => Some(Timedelta::NANOS_PER_DAY),
        "h" | "hr" | "hour" | "hours" => Some(Timedelta::NANOS_PER_HOUR),
        "m" | "min" | "minute" | "minutes" | "t" => Some(Timedelta::NANOS_PER_MIN),
        "s" | "sec" | "second" | "seconds" => Some(Timedelta::NANOS_PER_SEC),
        "ms" | "milli" | "millisecond" | "milliseconds" | "l" => Some(Timedelta::NANOS_PER_MILLI),
        "us" | "µs" | "micro" | "microsecond" | "microseconds" | "u" => Some(Timedelta::NANOS_PER_MICRO),
        "ns" | "nano" | "nanosecond" | "nanoseconds" | "n" => Some(1),
        _ => None,
    }
}
```

## 4. Arithmetic Operations

### 4.1 Timedelta + Timedelta

```rust
impl Scalar {
    pub fn add_timedelta(&self, other: &Scalar) -> Result<Scalar, TypeError> {
        match (self, other) {
            (Scalar::Timedelta64(a), Scalar::Timedelta64(b)) => {
                if *a == Timedelta::NAT || *b == Timedelta::NAT {
                    Ok(Scalar::Null(NullKind::NaT))
                } else {
                    a.checked_add(*b)
                        .map(Scalar::Timedelta64)
                        .ok_or_else(|| TypeError::Overflow)
                }
            }
            _ => Err(TypeError::InvalidOperation),
        }
    }
}
```

### 4.2 Timedelta * Scalar

```rust
pub fn mul_timedelta_scalar(&self, factor: f64) -> Result<Scalar, TypeError> {
    match self {
        Scalar::Timedelta64(nanos) if *nanos != Timedelta::NAT => {
            let result = (*nanos as f64 * factor).round() as i64;
            Ok(Scalar::Timedelta64(result))
        }
        Scalar::Timedelta64(_) => Ok(Scalar::Null(NullKind::NaT)),
        _ => Err(TypeError::InvalidOperation),
    }
}
```

### 4.3 Timedelta / Timedelta (Ratio)

```rust
pub fn div_timedelta(&self, other: &Scalar) -> Result<Scalar, TypeError> {
    match (self, other) {
        (Scalar::Timedelta64(a), Scalar::Timedelta64(b)) => {
            if *a == Timedelta::NAT || *b == Timedelta::NAT || *b == 0 {
                Ok(Scalar::Null(NullKind::NaN))
            } else {
                Ok(Scalar::Float64(*a as f64 / *b as f64))
            }
        }
        _ => Err(TypeError::InvalidOperation),
    }
}
```

## 5. Component Extraction

```rust
impl Timedelta {
    pub fn components(nanos: i64) -> TimedeltaComponents {
        if nanos == Self::NAT {
            return TimedeltaComponents::nat();
        }
        
        let negative = nanos < 0;
        let abs_nanos = nanos.unsigned_abs() as i64;
        
        let days = abs_nanos / Self::NANOS_PER_DAY;
        let rem = abs_nanos % Self::NANOS_PER_DAY;
        
        let hours = rem / Self::NANOS_PER_HOUR;
        let rem = rem % Self::NANOS_PER_HOUR;
        
        let minutes = rem / Self::NANOS_PER_MIN;
        let rem = rem % Self::NANOS_PER_MIN;
        
        let seconds = rem / Self::NANOS_PER_SEC;
        let rem = rem % Self::NANOS_PER_SEC;
        
        let milliseconds = rem / Self::NANOS_PER_MILLI;
        let rem = rem % Self::NANOS_PER_MILLI;
        
        let microseconds = rem / Self::NANOS_PER_MICRO;
        let nanoseconds = rem % Self::NANOS_PER_MICRO;
        
        TimedeltaComponents {
            days: if negative { -days } else { days },
            hours,
            minutes,
            seconds,
            milliseconds,
            microseconds,
            nanoseconds,
        }
    }
    
    pub fn total_seconds(nanos: i64) -> f64 {
        if nanos == Self::NAT {
            f64::NAN
        } else {
            nanos as f64 / Self::NANOS_PER_SEC as f64
        }
    }
}
```

## 6. Module-Level Function

### 6.1 to_timedelta() in fp-frame

```rust
/// Convert argument to Timedelta scalar or Series.
/// 
/// # Arguments
/// * `value` - String, numeric, or Series to convert
/// * `unit` - Unit for numeric values (default: "ns")
/// * `errors` - "raise", "coerce", or "ignore"
pub fn to_timedelta(value: &Scalar, unit: Option<&str>) -> Result<Scalar, FrameError> {
    match value {
        Scalar::Utf8(s) => {
            Timedelta::parse(s)
                .map(Scalar::Timedelta64)
                .map_err(|e| FrameError::CompatibilityRejected(e.to_string()))
        }
        Scalar::Int64(v) => {
            let multiplier = unit
                .and_then(unit_to_nanos)
                .unwrap_or(1);  // default ns
            Ok(Scalar::Timedelta64(v.saturating_mul(multiplier)))
        }
        Scalar::Float64(v) => {
            let multiplier = unit
                .and_then(unit_to_nanos)
                .unwrap_or(1) as f64;
            Ok(Scalar::Timedelta64((v * multiplier).round() as i64))
        }
        Scalar::Timedelta64(v) => Ok(Scalar::Timedelta64(*v)),
        Scalar::Null(_) => Ok(Scalar::Null(NullKind::NaT)),
        _ => Err(FrameError::CompatibilityRejected(
            "Cannot convert to timedelta".into()
        )),
    }
}

/// Series-level to_timedelta
pub fn to_timedelta_series(series: &Series, unit: Option<&str>, errors: &str) -> Result<Series, FrameError> {
    let values: Vec<Scalar> = series.values().iter().map(|v| {
        match to_timedelta(v, unit) {
            Ok(td) => td,
            Err(_) if errors == "coerce" => Scalar::Null(NullKind::NaT),
            Err(_) if errors == "ignore" => v.clone(),
            Err(e) => panic!("{e}"),  // errors="raise"
        }
    }).collect();
    
    Series::from_scalars(&series.name(), &values)
}
```

## 7. Column Integration (fp-columnar)

```rust
impl Column {
    /// Check if column contains Timedelta values
    pub fn is_timedelta(&self) -> bool {
        self.dtype() == DType::Timedelta64
    }
    
    /// Sum timedelta column (returns NaT if any NaT present)
    pub fn timedelta_sum(&self) -> Result<i64, ColumnError>;
    
    /// Mean timedelta column
    pub fn timedelta_mean(&self) -> Result<i64, ColumnError>;
}
```

## 8. Formatting

```rust
impl Timedelta {
    pub fn format(nanos: i64) -> String {
        if nanos == Self::NAT {
            return "NaT".to_string();
        }
        
        let comp = Self::components(nanos);
        let sign = if nanos < 0 { "-" } else { "" };
        
        // Format as "N days HH:MM:SS.fraction"
        let time_part = format!(
            "{:02}:{:02}:{:02}",
            comp.hours, comp.minutes, comp.seconds
        );
        
        let frac = comp.milliseconds * 1_000_000 
                 + comp.microseconds * 1_000 
                 + comp.nanoseconds;
        
        if frac > 0 {
            format!("{}{} days {}.{:09}", sign, comp.days.abs(), time_part, frac)
        } else {
            format!("{}{} days {}", sign, comp.days.abs(), time_part)
        }
    }
}
```

## 9. Implementation Phases

### Phase 1: Core (This PR)
1. Add `DType::Timedelta64` to fp-types
2. Add `Scalar::Timedelta64(i64)` variant
3. Implement `Timedelta` helper struct with constants and parsing
4. Basic string parsing ("1d", "2h 30m", "01:30:00")
5. Component extraction (days, seconds, total_seconds)
6. Arithmetic: add, sub, mul, div
7. Comparison operators
8. Formatting

### Phase 2: Series Integration
1. `to_timedelta()` module function
2. `to_timedelta_series()` for Series conversion
3. Series arithmetic with Timedelta
4. `.dt` accessor for Timedelta Series

### Phase 3: Index/Conformance
1. TimedeltaIndex support
2. `timedelta_range()` constructor
3. Conformance packet fixtures
4. Live oracle tests

## 10. Test Cases

```rust
#[test]
fn test_parse_simple() {
    assert_eq!(Timedelta::parse("1d"), Ok(86400 * 1_000_000_000));
    assert_eq!(Timedelta::parse("2h"), Ok(2 * 3600 * 1_000_000_000));
    assert_eq!(Timedelta::parse("30m"), Ok(30 * 60 * 1_000_000_000));
}

#[test]
fn test_parse_compound() {
    assert_eq!(
        Timedelta::parse("1d 2h 30m"),
        Ok((86400 + 2*3600 + 30*60) * 1_000_000_000)
    );
}

#[test]
fn test_components() {
    let nanos = 90061_001_002_003i64;  // 1d 1h 1m 1s 1ms 2us 3ns
    let comp = Timedelta::components(nanos);
    assert_eq!(comp.days, 1);
    assert_eq!(comp.hours, 1);
    assert_eq!(comp.minutes, 1);
    assert_eq!(comp.seconds, 1);
    assert_eq!(comp.milliseconds, 1);
    assert_eq!(comp.microseconds, 2);
    assert_eq!(comp.nanoseconds, 3);
}

#[test]
fn test_arithmetic() {
    let a = Scalar::Timedelta64(86400 * 1_000_000_000);  // 1 day
    let b = Scalar::Timedelta64(3600 * 1_000_000_000);   // 1 hour
    
    // Addition
    let sum = a.add_timedelta(&b).unwrap();
    assert_eq!(sum, Scalar::Timedelta64(90000 * 1_000_000_000));
}
```
