# DOC-PASS-04: Execution-Path Tracing and Control-Flow Narratives

**Bead:** bd-2gi.23.5
**Status:** Complete
**Date:** 2026-02-14
**Source Tree:** `legacy_pandas_code/pandas/pandas/` and `crates/fp-*/src/lib.rs`

---

## Table of Contents

1. [DataFrame Construction](#1-dataframe-construction)
2. [Column Selection](#2-column-selection)
3. [Binary Operations](#3-binary-operations)
4. [GroupBy Aggregation](#4-groupby-aggregation)
5. [Merge/Join](#5-mergejoin)
6. [CSV Read](#6-csv-read)
7. [loc/iloc Indexing](#7-lociloc-indexing)
8. [FrankenPandas Equivalences](#8-frankenpandas-equivalences)

---

## 1. DataFrame Construction

**Entry point:** `pd.DataFrame(data)` invokes `DataFrame.__init__` at `core/frame.py:455`.

### 1.1 Call Flow Overview

```
DataFrame.__init__(data, index, columns, dtype, copy)
    |
    +-- dtype validation: self._validate_dtype(dtype) if dtype is not None
    |
    +-- BRANCH on type(data):
    |     |
    |     +-- isinstance(data, DataFrame)  -->  extract data._mgr, shallow copy
    |     |     |
    |     |     +-- isinstance(data, BlockManager) + fastpath check (line 489)
    |     |         if index/columns/dtype all None and not copy:
    |     |           NDFrame.__init__(self, data)  --> RETURN (fastpath)
    |     |
    |     +-- isinstance(data, BlockManager)  -->  self._init_mgr(data, ...)
    |     |                                        core/generic.py
    |     |
    |     +-- isinstance(data, dict)  -->  dict_to_mgr(data, index, columns, ...)
    |     |                                core/internals/construction.py:375
    |     |
    |     +-- isinstance(data, ma.MaskedArray)
    |     |     +-- MaskedRecords  -->  raise TypeError
    |     |     +-- other  -->  sanitize_masked_array(data), then ndarray_to_mgr()
    |     |
    |     +-- isinstance(data, (np.ndarray, Series, Index, ExtensionArray))
    |     |     +-- data.dtype.names (structured array)  -->  rec_array_to_mgr()
    |     |     +-- Series/Index with non-None name     -->  dict_to_mgr({name: data}, ...)
    |     |     +-- other                               -->  ndarray_to_mgr()
    |     |
    |     +-- is_list_like(data)
    |     |     +-- has __array__ attr  -->  np.asarray(data)
    |     |     +-- not Sequence        -->  list(data)
    |     |     +-- len(data) > 0:
    |     |     |     +-- is_dataclass(data[0])  -->  dataclasses_to_dicts(data)
    |     |     |     +-- treat_as_nested(data)  -->  nested_data_to_arrays()
    |     |     |     |                              then arrays_to_mgr()
    |     |     |     +-- else                   -->  ndarray_to_mgr()
    |     |     +-- len(data) == 0              -->  dict_to_mgr({}, ...)
    |     |
    |     +-- scalar data
    |           +-- raise ValueError if index or columns is None
    |           +-- ExtensionDtype  -->  construct_1d_arraylike_from_scalar * N cols
    |           |                       then arrays_to_mgr()
    |           +-- else            -->  construct_2d_arraylike_from_scalar()
    |                                   then ndarray_to_mgr()
    |
    +-- NDFrame.__init__(self, mgr)    (line 656)
```

### 1.2 The dict_to_mgr Path (Most Common)

**File:** `core/internals/construction.py:375-460`

When `data` is a dict (the most common DataFrame construction pattern):

1. **Columns provided** (line 391): Iterate over `columns`, look up each in `data.keys()` using `data_keys.get_loc(column)`. Missing columns get `NaN` placeholders. Track missing-data column indices in `midxs` set.

2. **Columns not provided** (line 436): Use `data.keys()` directly as columns. Convert values via `com.maybe_iterable_to_list`.

3. **Index inference** (lines 413-423): If `index is None`, call `_extract_index(arrays)` which inspects the passed arrays to determine a suitable index. If any array is a Series, its index is used. Otherwise, a RangeIndex of the appropriate length is created.

4. **Copy semantics** (lines 441-458): Only EA (ExtensionArray) types are copied eagerly; numpy arrays will be copied during consolidation.

5. **Terminal call** (line 460): `arrays_to_mgr(arrays, columns, index, dtype=dtype, consolidate=copy)`.

### 1.3 The ndarray_to_mgr Path

**File:** `core/internals/construction.py:193-354`

Called when input is an ndarray, Series, Index, ExtensionArray, or list-of-scalars:

1. **Series handling** (lines 200-211): Extract `values`, `index`, `columns` from the Series. Reindex if `index` is explicitly provided and differs.

2. **1D-only EA branch** (lines 215-246): If dtype is 1D-only (e.g., Categorical, IntegerDtype), wrap as list-of-1D-arrays and delegate to `arrays_to_mgr`.

3. **Ensure 2D** (lines 248-277): For Series/Index, extract `._values` and reshape. For ndarrays, ensure `ndim == 2`. For other iterables, call `_prep_ndarraylike`.

4. **Dtype conversion** (lines 284-292): If `dtype` is explicit and values dtype differs, call `sanitize_array`.

5. **Shape validation** (line 299): `_check_values_indices_shape_match(values, index, columns)` ensures values shape matches `(len(index), len(columns))`.

6. **Transpose** (line 301): `values = values.T` (columns become the first dimension for block-level storage).

7. **Object type inference** (lines 306-346): If dtype is `object` and infer_object is True, attempt `lib.maybe_convert_objects` on each column to detect datetimelike data embedded in object arrays.

8. **Block creation** (line 344-346 or 323-329): Create `Block` objects via `new_block_2d` with appropriate `BlockPlacement`.

9. **Terminal call** (line 352): `create_block_manager_from_blocks(block_values, [columns, index])`.

### 1.4 The arrays_to_mgr Path

**File:** `core/internals/construction.py:96-152`

The common funnel for dict and nested-list construction:

1. **Integrity verification** (lines 110-118): If `verify_integrity=True`, extract index via `_extract_index(arrays)` and homogenize arrays via `_homogenize(arrays, index, dtype)`.

2. **Axes** (line 148): `axes = [columns, index]`.

3. **Terminal call** (line 150): `create_block_manager_from_column_arrays(arrays, axes, consolidate=consolidate, refs=refs)`.

This function (in `core/internals/managers.py`) groups arrays by dtype into homogeneous `Block` objects and creates a `BlockManager`.

### 1.5 Decision Points Summary

| Condition | Path | Terminal Function |
|-----------|------|-------------------|
| `data` is DataFrame | Extract `._mgr`, shallow copy | `NDFrame.__init__` (fastpath) |
| `data` is dict | `dict_to_mgr` | `arrays_to_mgr` -> `create_block_manager_from_column_arrays` |
| `data` is ndarray | `ndarray_to_mgr` | `create_block_manager_from_blocks` |
| `data` is list-of-dicts | `nested_data_to_arrays` | `arrays_to_mgr` |
| `data` is list-of-lists | `nested_data_to_arrays` or `ndarray_to_mgr` | depends on `treat_as_nested` |
| `data` is scalar | `construct_2d_arraylike_from_scalar` | `ndarray_to_mgr` |
| `data` is MaskedArray | `sanitize_masked_array` | `ndarray_to_mgr` |
| `data` is structured ndarray | `rec_array_to_mgr` | `arrays_to_mgr` |

---

## 2. Column Selection

**Entry point:** `df['col']` or `df[['col1', 'col2']]` invokes `DataFrame.__getitem__` at `core/frame.py:4162`.

### 2.1 Call Flow Overview

```
DataFrame.__getitem__(key)
    |
    +-- check_dict_or_set_indexers(key)   -- reject dict/set keys
    +-- key = lib.item_from_zerodim(key)  -- unwrap 0-d arrays
    +-- key = com.apply_if_callable(key, self)  -- resolve callables
    |
    +-- BRANCH 1: is_hashable(key) and not is_iterator(key)  (line 4167)
    |     |
    |     +-- Non-MultiIndex, key in columns (unique)
    |     |     RETURN self._get_item(key)  -->  Series  (line 4179)
    |     |
    |     +-- Non-MultiIndex, key in drop_duplicates(keep=False)
    |     |     RETURN self._get_item(key)  -->  Series
    |     |
    |     +-- MultiIndex, columns unique, key in columns
    |           RETURN self._getitem_multilevel(key)  (line 4182)
    |
    +-- BRANCH 2: isinstance(key, slice)  (line 4185)
    |     RETURN self._getitem_slice(key)  -->  DataFrame (row slice)
    |
    +-- BRANCH 3: isinstance(key, DataFrame)  (line 4189)
    |     RETURN self.where(key)  -->  DataFrame (boolean mask)
    |
    +-- BRANCH 4: com.is_bool_indexer(key)  (line 4193)
    |     RETURN self._getitem_bool_array(key)  -->  DataFrame (row filter)
    |
    +-- BRANCH 5: single key or collection  (line 4198)
          |
          +-- is_single_key (tuple or non-list-like)
          |     +-- columns.nlevels > 1  -->  _getitem_multilevel(key)
          |     +-- else: indexer = columns.get_loc(key)
          |             if integer  -->  indexer = [indexer]
          |
          +-- is_collection (list-like)
          |     indexer = columns._get_indexer_strict(key, "columns")[1]
          |
          +-- if indexer is boolean  -->  convert to int positions
          +-- if indexer is slice    -->  self._slice(indexer, axis=1)
          +-- else: data = self.take(indexer, axis=1)
                if single_key and data.shape[1] == 1 and not MultiIndex:
                    RETURN data._get_item(key)  -->  Series
                else:
                    RETURN data  -->  DataFrame
```

### 2.2 Single Column Selection (`df['col']`)

1. `is_hashable('col')` is True, enters BRANCH 1.
2. `self.columns.is_unique` and `'col' in self.columns` checks pass.
3. Calls `self._get_item('col')` (defined in `NDFrame`, `core/generic.py`).
4. This calls `self._mgr.iget(loc)` where `loc = self.columns.get_loc('col')`.
5. The BlockManager returns the values for that column's block position.
6. Wraps in a Series with `self.index` and name `'col'`.

**Performance note:** The unique-column fast-path avoids `drop_duplicates` (GH#45316).

### 2.3 Multi-Column Selection (`df[['col1', 'col2']]`)

1. `is_hashable(['col1', 'col2'])` is False (list is not hashable).
2. Falls through to BRANCH 5.
3. `is_list_like_indexer(['col1', 'col2'])` is True.
4. `indexer = self.columns._get_indexer_strict(['col1', 'col2'], "columns")[1]` returns integer positions.
5. `data = self.take(indexer, axis=1)` selects those columns.
6. `is_single_key` is False, so returns `data` as a DataFrame.

### 2.4 Boolean Indexing (`df[bool_series]`)

At `core/frame.py:4231`:

1. If `key` is a Series with different index, warn about reindexing.
2. If `key` length does not match `self.index` length, raise ValueError.
3. `key = check_bool_indexer(self.index, key)` -- align and validate.
4. If `key.all()`, return `self.copy(deep=False)` (fast path).
5. Otherwise, `indexer = key.nonzero()[0]`, then `self.take(indexer, axis=0)`.

### 2.5 MultiIndex Column Selection

At `core/frame.py:4258`:

1. `loc = self.columns.get_loc(key)` -- may return slice, ndarray, or int.
2. If slice/ndarray: select via `self.iloc[:, loc]`, drop levels from result columns.
3. Special handling for empty-string first-level (GH test_frame_getitem_multicolumn_empty_level).

---

## 3. Binary Operations

**Entry point:** `df + df2` invokes `OpsMixin.__add__` at `core/arraylike.py:101`.

### 3.1 Call Flow Overview

```
OpsMixin.__add__(self, other)
    |
    +-- @unpack_zerodim_and_defer("__add__")  (core/ops/common.py:30)
    |     +-- Check __pandas_priority__ for deference to higher-priority types
    |     +-- item_from_zerodim(other) -- unwrap 0-d ndarray
    |     +-- If other is list and self is not EA: wrap as Index/array
    |
    +-- Calls decorated method body: return self._arith_method(other, operator.add)
```

### 3.2 DataFrame._arith_method (core/frame.py:8960)

```
DataFrame._arith_method(self, other, op)
    |
    +-- _should_reindex_frame_op(other, op, axis=1, fill_value=None, level=None)
    |     Tests whether columns differ and require pre-alignment
    |     Conditions for reindex:
    |       - op is not pow/rpow (GH#32685)
    |       - other is DataFrame
    |       - MultiIndex columns differ, or
    |       - Column intersection differs from both sides' unique columns
    |     If True  -->  _arith_method_with_reindex(other, op)
    |
    +-- ops.maybe_prepare_scalar_for_op(other, shape)
    |     Wraps timedelta/Timedelta scalars, broadcasts
    |
    +-- self._align_for_op(other, axis=1, flex=True, level=None)
    |     (core/frame.py:9155)
    |     - For DataFrame other: align indexes, reindex to match
    |     - For Series other: align appropriately on axis
    |     - For ndarray/list/tuple: wrap as Series with appropriate axis labels
    |
    +-- self._dispatch_frame_op(other, op, axis=1)
    |     (core/frame.py:8975)
    |     +-- array_op = ops.get_array_op(op)
    |     |
    |     +-- BRANCH on type(right):
    |     |     +-- scalar (not list_like):
    |     |     |     bm = self._mgr.apply(array_op, right=right)
    |     |     |     Applies op to every block in the BlockManager
    |     |     |
    |     |     +-- DataFrame:
    |     |     |     assert indexes and columns match
    |     |     |     bm = self._mgr.operate_blockwise(right._mgr, array_op)
    |     |     |     Block-level paired operation
    |     |     |
    |     |     +-- Series (axis=1, column-wise):
    |     |     |     assert right.index equals self.columns
    |     |     |     Iterate column arrays, apply array_op per column
    |     |     |
    |     |     +-- Series (axis=0, row-wise):
    |     |           assert right.index equals self.index
    |     |           Broadcast right._values across all columns
    |
    +-- self._construct_result(new_data, other=other)
          Wrap result in DataFrame
```

### 3.3 Series._arith_method (core/series.py:6939)

```
Series._arith_method(self, other, op)
    |
    +-- self._align_for_op(other)  (core/series.py:6943)
    |     If other is Series and indexes differ:
    |       left, right = left.align(right)  -- union alignment
    |
    +-- base.IndexOpsMixin._arith_method(self, other, op)
          (core/base.py:1650)
          +-- Extract lvalues, rvalues from ._values
          +-- result = ops.arithmetic_op(lvalues, rvalues, op)
          +-- self._construct_result(result, name=res_name, other=other)
```

### 3.4 DataFrame._arith_method_with_reindex (core/frame.py:9065)

For DataFrames with non-matching columns:

1. Compute `cols = left.columns.intersection(right.columns)`.
2. Reindex both left and right to `cols`.
3. Perform block-wise operation on the intersected columns.
4. Reindex result to `left.columns.union(right.columns)`, filling new positions with NaN.

### 3.5 Alignment Flow (core/generic.py `NDFrame.align`)

The `align()` method computes the union (or intersection, depending on `join` parameter) of the two indexes, then reindexes both objects to the result. This is the core of pandas' "alignment-aware" computation.

---

## 4. GroupBy Aggregation

**Entry point:** `df.groupby('key').sum()` begins at `DataFrame.groupby` (`core/frame.py:12360`).

### 4.1 Call Flow Overview

```
DataFrame.groupby(by='key', sort=True, as_index=True, ...)
    |
    +-- from pandas.core.groupby import DataFrameGroupBy
    +-- RETURN DataFrameGroupBy(self, keys='key', sort=True, ...)
          |
          +-- GroupBy.__init__(obj, keys, ...)  (core/groupby/groupby.py:817)
                |
                +-- if grouper is None:
                |     grouper, exclusions, obj = get_grouper(
                |         obj, keys='key', level=None, sort=True, ...)
                |     (core/groupby/grouper.py:722)
                |
                +-- self._grouper = grouper  (BaseGrouper)
```

### 4.2 get_grouper (core/groupby/grouper.py:722)

```
get_grouper(obj, key='key', level=None, sort=True, ...)
    |
    +-- group_axis = obj.index
    |
    +-- BRANCH on level:
    |     +-- level is not None + MultiIndex  -->  extract level values
    |     +-- level is not None + single Index  -->  validate, set key=group_axis
    |
    +-- BRANCH on type(key):
    |     +-- isinstance(key, Grouper)  -->  key._get_grouper(obj)
    |     +-- isinstance(key, BaseGrouper)  -->  return directly
    |
    +-- Normalize: if not list, keys = [key]
    |
    +-- For each key in keys:
    |     Create Grouping object:
    |       - key is column name  -->  extract obj[key] as grouping values
    |       - key is callable     -->  apply to index
    |       - key is array-like   -->  use directly
    |
    +-- RETURN BaseGrouper(group_axis, groupings, sort=sort, ...)
```

The `BaseGrouper` computes `codes`, `uniques`, and `ngroups` from the grouping keys. The `codes` array maps each row to its group number.

### 4.3 GroupBy.sum() (core/groupby/groupby.py:2699)

```
GroupBy.sum(numeric_only=False, min_count=0, skipna=True, engine=None, ...)
    |
    +-- BRANCH on engine:
    |     +-- maybe_use_numba(engine)  -->  self._numba_agg_general(grouped_sum, ...)
    |     |     Uses JIT-compiled Numba kernel (line 2798)
    |     |
    |     +-- else (Cython path, default):
    |           with com.temp_setattr(self, "observed", True):
    |               result = self._agg_general(
    |                   numeric_only=..., min_count=...,
    |                   alias="sum", npfunc=np.sum, skipna=...)
    |
    +-- _agg_general (line 1444)
    |     RETURN self._cython_agg_general(how="sum", alt=np.sum, ...)
    |
    +-- _cython_agg_general (line 1510)
          |
          +-- data = self._get_data_to_aggregate(numeric_only=..., name="sum")
          |     (SeriesGroupBy: core/groupby/generic.py:194)
          |     (DataFrameGroupBy: core/groupby/generic.py:2902)
          |     Returns the BlockManager with only the relevant columns
          |
          +-- def array_func(values):
          |     TRY:
          |       result = self._grouper._cython_operation(
          |           "aggregate", values, "sum", ...)
          |       --> Calls Cython extension: _libs.groupby.group_sum()
          |     EXCEPT NotImplementedError:
          |       --> self._agg_py_fallback("sum", values, alt=np.sum)
          |           (Python-level per-group aggregation)
          |
          +-- new_mgr = data.grouped_reduce(array_func)
          |     Applies array_func to each block in the BlockManager
          |
          +-- res = self._wrap_agged_manager(new_mgr)
          +-- out = self._wrap_aggregated_output(res)
                Handles as_index, sort, column labeling
```

### 4.4 Path Selection Decision Points

| Condition | Path | Performance |
|-----------|------|-------------|
| `engine='numba'` | `_numba_agg_general` | JIT-compiled, good for large numeric data |
| Default (Cython), numeric dtype | `_cython_operation` -> `_libs.groupby.group_sum` | Fastest path, C-level loop |
| Non-numeric/EA dtype | `_agg_py_fallback` | Python loop per group, orders of magnitude slower |
| `numeric_only=True` | Filters to numeric blocks first | Reduces blocks to process |

### 4.5 The _agg_py_fallback Path (core/groupby/groupby.py:1462)

Triggered when `_cython_operation` raises `NotImplementedError` (e.g., for ExtensionArray dtypes or object columns):

1. Wrap values in a Series or single-column DataFrame.
2. Call `self._grouper.agg_series(ser, alt, preserve_dtype=True)`.
3. `agg_series` iterates through groups, applies `alt` (e.g., `np.sum`) to each group.
4. Ensure result dtype matches input dtype.
5. Reshape to match expected ndim via `ensure_block_shape`.

---

## 5. Merge/Join

**Entry point:** `pd.merge(left, right, on='key')` at `core/reshape/merge.py:146`.

### 5.1 Call Flow Overview

```
pd.merge(left, right, how='inner', on='key', ...)
    |
    +-- left_df = _validate_operand(left)   -- coerce Series to DataFrame
    +-- right_df = _validate_operand(right)
    |
    +-- BRANCH on how:
    |     +-- how == 'cross'  -->  _cross_merge(left_df, right_df, ...)
    |     +-- else:
    |           op = _MergeOperation(left_df, right_df, how='inner', on='key', ...)
    |           RETURN op.get_result()
```

### 5.2 _MergeOperation.__init__ (core/reshape/merge.py:961)

```
_MergeOperation.__init__(left, right, how, on, left_on, right_on, ...)
    |
    +-- self.how, self.anti_join = self._validate_how(how)
    |     Validates: left, right, inner, outer, left_anti, right_anti, cross, asof
    |     Anti-joins decompose to (left/right, anti_join=True)
    |
    +-- Validate left_index, right_index are bool
    +-- Check columns.nlevels match between left and right
    |
    +-- self.left_on, self.right_on = self._validate_left_right_on(left_on, right_on)
    |     Resolves 'on' into left_on/right_on lists
    |
    +-- self._get_merge_keys()  (line 1556)
    |     Extracts the actual join key arrays:
    |     - From column names: left[col], right[col]
    |     - From index: left.index, right.index
    |     - From array-like: direct use
    |     Returns: (left_join_keys, right_join_keys, join_names, left_drop, right_drop)
    |
    +-- Drop used index levels from left/right DataFrames
    +-- self._maybe_coerce_merge_keys()  -- dtype coercion for compatibility
    +-- validate uniqueness if validate is not None
```

### 5.3 _MergeOperation.get_result (core/reshape/merge.py:1134)

```
get_result()
    |
    +-- if indicator: _indicator_pre_merge(left, right)
    |     Adds _left_indicator and _right_indicator columns
    |
    +-- join_index, left_indexer, right_indexer = self._get_join_info()
    |     (core/reshape/merge.py:1402)
    |     |
    |     +-- BRANCH on index usage:
    |     |     +-- left_index AND right_index (both True):
    |     |     |     left_ax.join(right_ax, how=how, return_indexers=True)
    |     |     |     --> Uses Index.join() for sorted merge
    |     |     |
    |     |     +-- right_index AND how == 'left':
    |     |     |     _left_join_on_index(left_ax, right_ax, left_join_keys)
    |     |     |
    |     |     +-- left_index AND how == 'right':
    |     |     |     _left_join_on_index(right_ax, left_ax, right_join_keys)
    |     |     |
    |     |     +-- else (column-based join):
    |     |           left_indexer, right_indexer = self._get_join_indexers()
    |     |           --> get_join_indexers(left_keys, right_keys, sort, how)
    |     |
    |     +-- if anti_join: _handle_anti_join(join_index, left_indexer, right_indexer)
    |
    +-- result = self._reindex_and_concat(join_index, left_indexer, right_indexer)
    |     (line 1081)
    |     - Handle column suffix overlap
    |     - left._mgr.reindex_indexer(join_index, left_indexer, axis=1, ...)
    |     - right._mgr.reindex_indexer(join_index, right_indexer, axis=1, ...)
    |     - concat([left, right], axis=1)
    |
    +-- if indicator: _indicator_post_merge(result)
    +-- _maybe_add_join_keys(result, ...)
    +-- _maybe_restore_index_levels(result)
    +-- RETURN result.__finalize__(...)
```

### 5.4 get_join_indexers -- The Hash vs Sort-Merge Decision

**File:** `core/reshape/merge.py:2043`

```
get_join_indexers(left_keys, right_keys, sort, how)
    |
    +-- Empty fast-paths:
    |     left_n == 0 and how in [left, inner]  -->  empty result
    |     right_n == 0 and how in [right, inner]  -->  empty result
    |
    +-- BRANCH on number of join keys:
    |     +-- len(left_keys) > 1 (multi-column join):
    |     |     Factorize each key pair: _factorize_keys(left_keys[n], right_keys[n])
    |     |     Flatten into composite i8 keys via _get_join_keys(llab, rlab, shape)
    |     |
    |     +-- len(left_keys) == 1 (single-column join):
    |           lkey = left_keys[0], rkey = right_keys[0]
    |
    +-- left = Index(lkey), right = Index(rkey)
    |
    +-- DECISION POINT (line 2103):
    |     IF left.is_monotonic_increasing
    |        AND right.is_monotonic_increasing
    |        AND (left.is_unique OR right.is_unique):
    |       --> SORT-MERGE path:
    |           _, lidx, ridx = left.join(right, how=how, return_indexers=True)
    |           Uses binary search / merge-based join in _libs.join
    |
    |     ELSE:
    |       --> HASH JOIN path:
    |           get_join_indexers_non_unique(left._values, right._values, sort, how)
    |           (line 2121)
    |           1. _factorize_keys(left, right)  -- build hash table, get integer codes
    |           2. Call libjoin.left_outer_join / inner_join / full_outer_join
    |              (Cython implementations in _libs/join.pyx)
    |
    +-- Optimize: if indexer is identity (range_indexer), set to None
```

### 5.5 Sort-Merge vs Hash Join Decision Summary

| Condition | Join Algorithm | Complexity |
|-----------|---------------|-----------|
| Both keys sorted, at least one unique | Sort-merge via `Index.join` | O(n + m) |
| Keys not sorted or both non-unique | Hash join via `_factorize_keys` + `libjoin` | O(n + m) average |
| Multi-column keys | Factorize + flatten first, then same decision | Extra O(n * k) for factorization |

---

## 6. CSV Read

**Entry point:** `pd.read_csv('file.csv')` at `io/parsers/readers.py:350`.

### 6.1 Call Flow Overview

```
pd.read_csv(filepath_or_buffer, sep=..., engine=..., ...)
    |
    +-- Collect locals() into kwds dict  (line 857)
    +-- kwds_defaults = _refine_defaults_read(dialect, delimiter, engine, sep, ...)
    |     Resolves engine selection:
    |     - If sep is regex or multi-char  -->  force 'python' engine
    |     - If skipfooter > 0              -->  force 'python' engine
    |     - If engine not specified         -->  default to 'c' (for read_csv)
    |
    +-- RETURN _read(filepath_or_buffer, kwds)
```

### 6.2 _read() (io/parsers/readers.py:258)

```
_read(filepath_or_buffer, kwds)
    |
    +-- Resolve parse_dates default (line 264)
    +-- Extract iterator, chunksize flags
    +-- Validate encoding_errors is str
    |
    +-- BRANCH on engine == 'pyarrow':
    |     Reject iterator and chunksize options
    |
    +-- chunksize = validate_integer("chunksize", chunksize, 1)
    +-- _validate_names(names)
    |
    +-- parser = TextFileReader(filepath_or_buffer, **kwds)
    |     (io/parsers/readers.py:1593)
    |
    +-- BRANCH on chunksize or iterator:
    |     +-- True   -->  RETURN parser  (lazy TextFileReader)
    |     +-- False  -->  with parser: RETURN parser.read(nrows)
```

### 6.3 TextFileReader.__init__ (io/parsers/readers.py:1600)

```
TextFileReader.__init__(f, engine=None, **kwds)
    |
    +-- Engine resolution:
    |     If engine not specified: engine = 'python'  (overridden by _refine_defaults)
    |
    +-- _validate_skipfooter(kwds)
    +-- Dialect handling: _extract_dialect, _merge_with_dialect_properties
    +-- Header inference: header = 0 if names is None else None
    |
    +-- options = self._get_options_with_defaults(engine)
    |     Merges parser_defaults, _c_parser_defaults, _fwf_defaults
    |     Validates engine-specific option compatibility
    |
    +-- self.options, self.engine = self._clean_options(options, engine)
    |     Further engine refinement:
    |     - If engine == 'c' and python-only features needed, fall back
    |     - If skipfooter > 0 and engine == 'c', switch to 'python'
    |
    +-- self._engine = self._make_engine(f, self.engine)
```

### 6.4 _make_engine -- Engine Selection (io/parsers/readers.py:1871)

```
_make_engine(f, engine)
    |
    +-- Engine mapping:
    |     "c"          --> CParserWrapper      (fastest, C extension)
    |     "python"     --> PythonParser         (pure Python, most features)
    |     "pyarrow"    --> ArrowParserWrapper   (Arrow-based, columnar)
    |     "python-fwf" --> FixedWidthFieldParser
    |
    +-- File handle opening:
    |     +-- pyarrow: binary mode ("rb")
    |     +-- c engine + utf-8: binary mode ("rb") -- c engine decodes utf-8 internally
    |     +-- else: text mode ("r")
    |
    +-- self.handles = get_handle(f, mode, encoding=..., compression=..., memory_map=...)
    +-- RETURN mapping[engine](f, **self.options)
```

### 6.5 TextFileReader.read() (io/parsers/readers.py:1931)

```
TextFileReader.read(nrows=None)
    |
    +-- BRANCH on engine:
    |     +-- 'pyarrow':
    |     |     df = self._engine.read()  -- returns DataFrame directly
    |     |
    |     +-- 'c' or 'python':
    |           index, columns, col_dict = self._engine.read(nrows)
    |           |
    |           +-- CParserWrapper.read():
    |           |     Calls C-level parser (_libs.parsers.TextReader)
    |           |     Returns (index_obj, column_names, {col_name: ndarray})
    |           |
    |           +-- PythonParser.read():
    |                 Pure Python tokenizer, handles edge cases
    |                 Returns same (index, columns, col_dict) tuple
    |
    +-- Index creation (line 1954):
    |     If index is None: RangeIndex(currow, currow + new_rows)
    |
    +-- dtype enforcement (line 1969-1990):
    |     Apply user-specified dtype dict to each column via Series construction
    |
    +-- df = DataFrame(col_dict, columns=columns, index=index, copy=False)
    |     Goes through dict_to_mgr path (section 1.2)
    |
    +-- RETURN df
```

### 6.6 Engine Selection Decision Summary

| Condition | Engine | Notes |
|-----------|--------|-------|
| Default, no special options | `c` (CParserWrapper) | Fastest for typical CSV files |
| `engine='python'` explicit | `python` (PythonParser) | Full feature support |
| `engine='pyarrow'` explicit | `pyarrow` (ArrowParserWrapper) | Columnar, no chunking |
| Multi-char separator (len > 1) | `python` (auto-fallback) | C parser requires single-char sep |
| Regex separator | `python` (auto-fallback) | C parser does not support regex |
| `skipfooter > 0` | `python` (auto-fallback) | C parser does not support skipfooter |
| `sep=None` (sniffing) | `python` (auto-fallback) | Delimiter sniffing not in C parser |

---

## 7. loc/iloc Indexing

### 7.1 loc Indexing

**Entry point:** `df.loc[rows, cols]` invokes `_LocIndexer.__getitem__` at `core/indexing.py:1190`.

```
_LocationIndexer.__getitem__(key)  (line 1190)
    |
    +-- check_dict_or_set_indexers(key)
    +-- BRANCH on type(key):
    |
    +-- tuple key (e.g., df.loc[row, col]):
    |     +-- Apply callable elements
    |     +-- _is_scalar_access(key)?  (line 1608)
    |     |     Checks: len(key) == ndim, all scalars, no MultiIndex, unique axes
    |     |     If True  -->  obj._get_value(*key, takeable=False)
    |     |                   Direct scalar access (fastest path)
    |     |
    |     +-- _getitem_tuple(key)  (line 1722)
    |           +-- TRY: _getitem_lowerdim(tup)
    |           |     Attempts to reduce dimensionality (DataFrame -> Series)
    |           |
    |           +-- _multi_take_opportunity(tup)?
    |           |     If all elements are list-like and none are boolean:
    |           |     --> _multi_take(tup)
    |           |         Compute indexers for both axes at once, then reindex
    |           |
    |           +-- _getitem_tuple_same_dim(tup)
    |                 Process each axis sequentially
    |
    +-- non-tuple key (e.g., df.loc[rows]):
          axis = 0
          +-- _getitem_axis(key, axis=0)  (line 1754)
```

### 7.2 _LocIndexer._getitem_axis (core/indexing.py:1754)

```
_LocIndexer._getitem_axis(key, axis)
    |
    +-- key = item_from_zerodim(key)
    +-- Handle Ellipsis: key = slice(None)
    |
    +-- BRANCH on type(key):
    |
    +-- isinstance(key, slice):
    |     _get_slice_axis(key, axis)
    |     --> labels.slice_indexer(start, stop, step)
    |     --> self.obj._slice(indexer, axis)
    |     Label-based slicing (INCLUSIVE of both endpoints)
    |
    +-- com.is_bool_indexer(key):
    |     _getbool_axis(key, axis)
    |     --> check_bool_indexer(labels, key)
    |     --> key.nonzero()[0]
    |     --> self.obj.take(inds, axis)
    |
    +-- is_list_like_indexer(key):
    |     +-- Nested tuple + MultiIndex  -->  labels.get_locs(key)
    |     +-- Otherwise:
    |           _getitem_iterable(key, axis)
    |           --> _get_listlike_indexer(key, axis)
    |           --> obj._reindex_with_indexers({axis: [keyarr, indexer]})
    |
    +-- scalar key (fall-through):
          _validate_key(key, axis)
          _get_label(key, axis)
          --> self.obj.xs(label, axis)  -- cross-section
```

### 7.3 iloc Indexing

**Entry point:** `df.iloc[rows, cols]` invokes `_iLocIndexer.__getitem__` at `core/indexing.py:1190` (shared base).

The flow is similar to loc but uses positional (integer-based) indexing:

```
_iLocIndexer._getitem_axis(key, axis)  (line 2221)
    |
    +-- Handle Ellipsis: key = slice(None)
    +-- Reject DataFrame key (must use .loc for alignment)
    |
    +-- BRANCH on type(key):
    |
    +-- isinstance(key, slice):
    |     _get_slice_axis(key, axis)
    |     --> labels._validate_positional_slice(slice_obj)
    |     --> self.obj._slice(slice_obj, axis)
    |     Standard Python slicing semantics (EXCLUSIVE of stop)
    |
    +-- isinstance(key, list):
    |     key = np.asarray(key)
    |     Falls through to is_list_like_indexer
    |
    +-- com.is_bool_indexer(key):
    |     _getbool_axis(key, axis)
    |
    +-- is_list_like_indexer(key):
    |     _get_list_axis(key, axis)
    |     --> self.obj.take(key, axis)
    |     Direct positional take
    |
    +-- scalar integer:
          _validate_integer(key, axis)  -- bounds check
          self.obj._ixs(key, axis)
          --> axis=0: return row as Series
          --> axis=1: return column as Series
```

### 7.4 Key Behavioral Differences: loc vs iloc

| Aspect | loc | iloc |
|--------|-----|------|
| Key type | Labels | Integer positions |
| Slice semantics | Inclusive of both endpoints | Exclusive of stop (Python convention) |
| Boolean mask | Aligned to index first | Must match exact length |
| List key | Labels -> `_get_listlike_indexer` | Positions -> `take` |
| Scalar key | `xs(label)` | `_ixs(position)` |
| Out-of-bounds | KeyError | IndexError |
| MultiIndex | Partial indexing supported | No partial indexing |

### 7.5 _is_scalar_access Fast Path

Both loc and iloc implement `_is_scalar_access` to detect `df.loc[row, col]` with scalar keys and unique axes. When triggered, this bypasses all the branching logic and calls `obj._get_value(row, col)` directly, which extracts a single scalar from the BlockManager without constructing intermediate Series/DataFrames.

- **loc** (line 1608): Requires `len(key) == ndim`, all scalars, no MultiIndex, no partial string indexing, unique axes.
- **iloc** (line 2154): Requires `len(key) == ndim`, all integers.

---

## 8. FrankenPandas Equivalences

### 8.1 DataFrame Construction

**Pandas:** `pd.DataFrame(data)` -> `dict_to_mgr`/`ndarray_to_mgr` -> `BlockManager`
**FrankenPandas:** `DataFrame::from_dict()` or `DataFrame::from_series()`

| Pandas Path | FP Equivalent | File | Key Differences |
|-------------|---------------|------|-----------------|
| `dict_to_mgr` | `DataFrame::from_dict(column_order, data)` | `fp-frame/src/lib.rs:674` | No BlockManager; `BTreeMap<String, Column>` directly. No dtype coercion; `Scalar` enum carries type. No consolidation needed. |
| `ndarray_to_mgr` | No direct equivalent | -- | FP does not accept raw ndarrays; data enters as `Vec<Scalar>`. |
| `arrays_to_mgr` | `DataFrame::new(index, columns)` | `fp-frame/src/lib.rs:623` | Validates `column.len() == index.len()` but no block grouping. |
| `DataFrame(other_df)` | `DataFrame::clone()` | -- | Rust `Clone` trait. No shallow-copy/CoW semantics. |
| `from_series` | `DataFrame::from_series(series_list)` | `fp-frame/src/lib.rs:639` | Pre-computes N-way union index, then reindexes each series once (O(N) vs pandas O(N^2) iterative alignment). |

**Structural divergence:** Pandas uses a `BlockManager` that groups columns by dtype into 2D blocks with `_blknos`/`_blklocs` indirection. FP uses a flat `BTreeMap<String, Column>` where each column is independently stored. This eliminates consolidation overhead but may reduce cache locality for homogeneous-dtype operations.

### 8.2 Column Selection

**Pandas:** `df['col']` -> `__getitem__` -> `_get_item` -> `BlockManager.iget(loc)`
**FrankenPandas:** `df.column("col")` returns `Option<&Column>`

| Pandas Path | FP Equivalent | File | Differences |
|-------------|---------------|------|-------------|
| Single column `df['col']` | `df.column("col")` | `fp-frame/src/lib.rs:792` | Returns `Option<&Column>` (borrow), not a new Series. No view/copy semantics. |
| Multi-column `df[['a','b']]` | `df.select_columns(&["a","b"])` | `fp-frame/src/lib.rs:738` | Returns new DataFrame with cloned columns. Error if column missing. |
| Boolean `df[mask]` | `df.filter_rows(&mask_series)` | `fp-frame/src/lib.rs:801` | Aligns mask index first, then filters. Returns new DataFrame. |

### 8.3 Binary Operations

**Pandas:** `s1 + s2` -> `__add__` -> `_arith_method` -> `_align_for_op` -> `align()` -> element-wise op
**FrankenPandas:** `s1.add(&s2)` -> `binary_op_with_policy` -> `align_union` -> `Column::binary_numeric`

| Pandas Step | FP Equivalent | Differences |
|-------------|---------------|-------------|
| `unpack_zerodim_and_defer` | Not needed | No 0-d arrays in Rust. |
| `_align_for_op` -> `align()` | `align_union(&self.index, &other.index)` | FP always does outer alignment. Uses `AlignmentPlan` with position maps. |
| `ops.arithmetic_op(lvalues, rvalues, op)` | `Column::binary_numeric(&left, &right, op)` | FP operates on `Vec<Scalar>` with NA propagation per `Scalar::is_missing()`. |
| `_construct_result` | `Series::new(name, plan.union_index, column)` | FP builds result directly; no BlockManager reconstruction. |
| Copy-on-Write tracking | N/A | FP always creates new data (Rust ownership model). |
| **RuntimePolicy check** | `policy.decide_join_admission()` | FP adds conformance checking: strict mode can reject duplicate-label alignment and large-cardinality joins. Pandas has no equivalent. |

**Performance note:** FP's `align_union` uses a borrowed-key `HashMap` (AG-02 optimization) to avoid cloning index labels during alignment. Pandas uses `Index.join()` which may involve Cython-level merge.

### 8.4 GroupBy Aggregation

**Pandas:** `df.groupby('key').sum()` -> `get_grouper` -> `_cython_agg_general` -> `_libs.groupby.group_sum`
**FrankenPandas:** `groupby_sum(keys, values, options, policy, ledger)`

| Pandas Step | FP Equivalent | Differences |
|-------------|---------------|-------------|
| `get_grouper()` | Caller provides key/value Series directly | FP has no lazy grouper object. Grouping is eager. |
| `DataFrameGroupBy` object | No object; free function `groupby_sum()` | FP does not support method chaining on groups. |
| `_cython_agg_general` | `groupby_sum_with_trace()` | FP dispatches to dense-int64 or HashMap path. |
| Dense int64 bucket path | `try_groupby_sum_dense_int64()` | FP detects contiguous i64 key ranges and uses array-indexed sums. |
| Cython `group_sum` | HashMap accumulation | FP uses `HashMap<GroupKeyRef, (usize, f64)>` with borrowed keys (AG-08). |
| Python fallback | No fallback needed | All FP types handled uniformly via `Scalar`. |
| Arena allocation | `groupby_sum_with_arena()` | FP uses `bumpalo::Bump` arena when estimated intermediate size fits budget (default 256MB). |
| Numba path | Not implemented | FP relies on Rust compiler optimizations instead. |

**Architectural note:** FP's groupby is a set of free functions (`groupby_sum`, `groupby_mean`, `groupby_min`, `groupby_max`, `groupby_count`), each implementing the full pipeline. Pandas has a single `_cython_agg_general` dispatch that selects the Cython kernel by name.

### 8.5 Merge/Join

**Pandas:** `pd.merge(left, right, on='key')` -> `_MergeOperation` -> `get_join_indexers` -> `libjoin`
**FrankenPandas:** `join_series(left, right, join_type)` or `merge_dataframes()`

| Pandas Step | FP Equivalent | Differences |
|-------------|---------------|-------------|
| `_MergeOperation.__init__` | `join_series_with_options()` | FP takes Series pair + JoinType enum. No multi-key join yet. |
| `_get_merge_keys()` | Caller extracts keys | FP does not auto-detect join keys from column names. |
| Sort-merge path (`Index.join`) | Not implemented separately | FP always uses hash join. |
| Hash join (`_factorize_keys` + `libjoin`) | `HashMap<&IndexLabel, Vec<usize>>` | FP builds right-side hash map with borrowed keys (AG-02). For Right/Outer, also builds left-side map. |
| `_reindex_and_concat` | Direct output construction | FP builds output index + position arrays, then reindexes both columns in one pass. |
| Anti-join support | Not implemented | Pandas supports `left_anti`/`right_anti` since 3.0. |
| Arena allocation | `join_series_with_arena()` | FP uses `bumpalo::Bump` arena for output vectors when estimated size fits budget. |
| Cross merge | Not implemented | Would require Cartesian product generation. |

### 8.6 CSV Read

**Pandas:** `pd.read_csv('file.csv')` -> `_read` -> `TextFileReader` -> `CParserWrapper` or `PythonParser`
**FrankenPandas:** `fp_io::read_csv(path)` or `fp_io::read_csv_str(input)`

| Pandas Step | FP Equivalent | Differences |
|-------------|---------------|-------------|
| Engine selection (C/Python/PyArrow) | Single Rust implementation | FP uses the `csv` crate (pure Rust). No engine selection needed. |
| `CParserWrapper` (C extension) | `read_csv_str()` using `csv::ReaderBuilder` | FP processes all in memory. No chunking support. |
| `PythonParser` (fallback) | Not needed | Rust parser handles all features directly. |
| dtype inference | `parse_scalar()` | FP tries i64, f64, bool in order; falls back to Utf8. No datetime inference. |
| Memory mapping | Not supported | FP reads entire file to string first. |
| Chunked reading | Not supported | FP reads all rows at once. |
| NA value handling | `parse_scalar_with_na()` with `CsvReadOptions.na_values` | User-specified list of NA strings. No `keep_default_na`. |
| Index column | `CsvReadOptions.index_col` | FP supports extracting one column as index. No multi-column index. |
| Custom delimiter | `CsvReadOptions.delimiter` | FP supports single-byte delimiter via `csv` crate. |
| Compression | Not supported | FP reads plain text only. |

### 8.7 Indexing (loc/iloc)

**Pandas:** `df.loc[rows, cols]` -> `_LocIndexer.__getitem__` -> `_getitem_axis` -> label-based lookup
**FrankenPandas:** No dedicated loc/iloc indexer objects.

| Pandas Feature | FP Equivalent | Differences |
|----------------|---------------|-------------|
| `df.loc[label]` | `index.position(&label)` then manual extraction | No `_LocIndexer` object. Position lookup is O(log n) for sorted, O(n) for unsorted (AG-13 adaptive backend). |
| `df.iloc[i]` | Direct index into `column.values()[i]` | No bounds checking wrapper. |
| Boolean mask `df.loc[mask]` | `series.filter(&mask)` / `df.filter_rows(&mask)` | Aligns mask index first. |
| Slice `df.loc['a':'c']` | No direct equivalent | Would need `Index::slice_by_labels()`. |
| Label list `df.loc[['a','b']]` | `series.reindex(labels)` | Creates new Series/DataFrame via reindex. |
| Scalar access `df.loc[r, c]` | `df.column(c)?.value(index.position(&r)?)` | Manual two-step: find column, then find row position. |
| `_is_scalar_access` fast path | `Index::position()` with binary search | FP's sorted-index binary search (AG-13) is analogous to pandas' scalar fast path. |

---

## Appendix A: File Reference Index

### Pandas Source Files

| File | Key Functions | Section |
|------|---------------|---------|
| `core/frame.py:455` | `DataFrame.__init__` | 1 |
| `core/frame.py:4162` | `DataFrame.__getitem__` | 2 |
| `core/frame.py:8960` | `DataFrame._arith_method` | 3 |
| `core/frame.py:9155` | `DataFrame._align_for_op` | 3 |
| `core/frame.py:12360` | `DataFrame.groupby` | 4 |
| `core/internals/construction.py:96` | `arrays_to_mgr` | 1 |
| `core/internals/construction.py:193` | `ndarray_to_mgr` | 1 |
| `core/internals/construction.py:375` | `dict_to_mgr` | 1 |
| `core/arraylike.py:101` | `OpsMixin.__add__` | 3 |
| `core/ops/common.py:30` | `unpack_zerodim_and_defer` | 3 |
| `core/series.py:6939` | `Series._arith_method` | 3 |
| `core/series.py:6943` | `Series._align_for_op` | 3 |
| `core/groupby/groupby.py:746` | `class GroupBy` | 4 |
| `core/groupby/groupby.py:817` | `GroupBy.__init__` | 4 |
| `core/groupby/groupby.py:1444` | `GroupBy._agg_general` | 4 |
| `core/groupby/groupby.py:1462` | `GroupBy._agg_py_fallback` | 4 |
| `core/groupby/groupby.py:1510` | `GroupBy._cython_agg_general` | 4 |
| `core/groupby/groupby.py:2699` | `GroupBy.sum` | 4 |
| `core/groupby/grouper.py:722` | `get_grouper` | 4 |
| `core/reshape/merge.py:146` | `merge()` | 5 |
| `core/reshape/merge.py:938` | `class _MergeOperation` | 5 |
| `core/reshape/merge.py:1134` | `_MergeOperation.get_result` | 5 |
| `core/reshape/merge.py:1391` | `_MergeOperation._get_join_indexers` | 5 |
| `core/reshape/merge.py:2043` | `get_join_indexers` | 5 |
| `core/reshape/merge.py:2121` | `get_join_indexers_non_unique` | 5 |
| `io/parsers/readers.py:258` | `_read` | 6 |
| `io/parsers/readers.py:350` | `read_csv` | 6 |
| `io/parsers/readers.py:1593` | `class TextFileReader` | 6 |
| `io/parsers/readers.py:1871` | `TextFileReader._make_engine` | 6 |
| `io/parsers/readers.py:1931` | `TextFileReader.read` | 6 |
| `core/indexing.py:1190` | `_LocationIndexer.__getitem__` | 7 |
| `core/indexing.py:1227` | `class _LocIndexer` | 7 |
| `core/indexing.py:1608` | `_LocIndexer._is_scalar_access` | 7 |
| `core/indexing.py:1722` | `_LocIndexer._getitem_tuple` | 7 |
| `core/indexing.py:1754` | `_LocIndexer._getitem_axis` | 7 |
| `core/indexing.py:1920` | `class _iLocIndexer` | 7 |
| `core/indexing.py:2154` | `_iLocIndexer._is_scalar_access` | 7 |
| `core/indexing.py:2221` | `_iLocIndexer._getitem_axis` | 7 |

### FrankenPandas Source Files

| File | Key Functions/Structs | Section |
|------|----------------------|---------|
| `crates/fp-frame/src/lib.rs:29` | `struct Series` | 8.1-8.3 |
| `crates/fp-frame/src/lib.rs:107` | `Series::binary_op_with_policy` | 8.3 |
| `crates/fp-frame/src/lib.rs:617` | `struct DataFrame` | 8.1 |
| `crates/fp-frame/src/lib.rs:623` | `DataFrame::new` | 8.1 |
| `crates/fp-frame/src/lib.rs:639` | `DataFrame::from_series` | 8.1 |
| `crates/fp-frame/src/lib.rs:674` | `DataFrame::from_dict` | 8.1 |
| `crates/fp-frame/src/lib.rs:738` | `DataFrame::select_columns` | 8.2 |
| `crates/fp-frame/src/lib.rs:801` | `DataFrame::filter_rows` | 8.2 |
| `crates/fp-index/src/lib.rs:108` | `struct Index` | 8.7 |
| `crates/fp-index/src/lib.rs:196` | `Index::position` (adaptive lookup) | 8.7 |
| `crates/fp-groupby/src/lib.rs:58` | `groupby_sum` | 8.4 |
| `crates/fp-groupby/src/lib.rs:88` | `groupby_sum_with_trace` | 8.4 |
| `crates/fp-join/src/lib.rs:58` | `join_series` | 8.5 |
| `crates/fp-join/src/lib.rs:76` | `join_series_with_trace` | 8.5 |
| `crates/fp-io/src/lib.rs:59` | `read_csv_str` | 8.6 |
| `crates/fp-io/src/lib.rs:175` | `read_csv_with_options` | 8.6 |

---

## Appendix B: Cross-Cutting Concerns

### B.1 Copy-on-Write (CoW) in Pandas

Pandas 3.0 enforces Copy-on-Write semantics throughout the codebase. Every `Block` tracks references via `BlockValuesRefs`, and mutations trigger lazy copies. This affects all seven workflows:

- **Construction:** `BlockManager.copy(deep=False)` creates shallow copies with shared refs.
- **Column selection:** `_get_item` returns a view; mutations trigger CoW.
- **Binary ops:** Result is always a new DataFrame (no CoW interaction).
- **GroupBy:** aggregated results are always new.
- **Merge:** result is always new; intermediate reindexing uses `only_slice=True`.
- **CSV read:** `copy=False` passed to DataFrame constructor.
- **Indexing:** `.loc`/`.iloc` return views for slices, copies for fancy indexing.

FrankenPandas does not implement CoW. Rust's ownership model ensures that data is either moved or explicitly cloned, making CoW unnecessary.

### B.2 Error Propagation Patterns

**Pandas:** Raises Python exceptions (`KeyError`, `ValueError`, `IndexError`, `TypeError`) with detailed messages. Many error paths include GH issue references. Errors can be deeply nested in the call stack.

**FrankenPandas:** Uses `Result<T, Error>` types with `thiserror`-derived error enums. Errors propagate via `?` operator. No exception unwinding. Error types include:
- `FrameError` (length mismatch, duplicate index, compatibility rejection)
- `ColumnError` (type mismatch, out of bounds)
- `IndexError` (validation failures)
- `GroupByError`, `JoinError`, `IoError` (domain-specific wrappers)

### B.3 NA/Missing Value Handling

**Pandas:** Uses `np.nan` for float, `pd.NA` for nullable integer/boolean/string, `pd.NaT` for datetime. NA propagation rules differ by dtype and operation.

**FrankenPandas:** Uses `Scalar::Null(NullKind)` with variants `Null`, `NaN`, `NaT`. All operations check `scalar.is_missing()` uniformly. No implicit type-based NA selection.
