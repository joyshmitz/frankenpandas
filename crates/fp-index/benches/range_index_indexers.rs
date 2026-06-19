use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fp_index::RangeIndex;

const SIZES: &[usize] = &[100_000, 1_000_000];

fn build_source(size: usize) -> RangeIndex {
    RangeIndex::new(0, (size as i64) * 2, 2).expect("valid source range")
}

fn build_miss_heavy_targets(size: usize) -> Vec<i64> {
    (0..size)
        .map(|i| {
            if i % 16 == 0 {
                ((i % size) as i64) * 2
            } else {
                ((i as i64) * 2) + 1
            }
        })
        .collect()
}

fn build_all_miss_target_range(size: usize) -> RangeIndex {
    RangeIndex::new(1, (size as i64) * 2 + 1, 2).expect("valid target range")
}

fn build_all_miss_targets(size: usize) -> Vec<i64> {
    (0..size).map(|i| (i as i64) * 2 + 1).collect()
}

fn checksum_indexer(indexer: &[isize]) -> isize {
    indexer.iter().fold(indexer.len() as isize, |acc, value| {
        acc.wrapping_add(*value)
    })
}

fn current_get_indexer_checksum(source: &RangeIndex, targets: &[i64]) -> isize {
    checksum_indexer(&source.get_indexer(targets))
}

fn legacy_get_loc_indexer_checksum(source: &RangeIndex, targets: &[i64]) -> isize {
    let mut checksum = targets.len() as isize;
    for &target in targets {
        let position = source
            .get_loc(target)
            .map_or(-1, |position| position as isize);
        checksum = checksum.wrapping_add(position);
    }
    checksum
}

fn current_reindex_checksum(source: &RangeIndex, target: &RangeIndex) -> isize {
    let (_, indexer) = source.reindex(target);
    checksum_indexer(&indexer)
}

fn legacy_get_loc_reindex_checksum(source: &RangeIndex, targets: &[i64]) -> isize {
    let mut checksum = targets.len() as isize;
    for &value in targets {
        let source_position = source
            .get_loc(value)
            .map_or(-1, |position| position as isize);
        checksum = checksum.wrapping_add(source_position);
    }
    checksum
}

fn bench_range_index_indexers(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_index_indexers");
    for &size in SIZES {
        let source = build_source(size);
        let targets = build_miss_heavy_targets(size);
        let target_range = build_all_miss_target_range(size);
        let target_range_values = build_all_miss_targets(size);

        group.bench_with_input(
            BenchmarkId::new("current_get_indexer_miss_heavy", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(current_get_indexer_checksum(
                        black_box(&source),
                        black_box(&targets),
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("legacy_get_loc_loop_miss_heavy", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(legacy_get_loc_indexer_checksum(
                        black_box(&source),
                        black_box(&targets),
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("current_reindex_all_miss", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(current_reindex_checksum(
                        black_box(&source),
                        black_box(&target_range),
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("legacy_get_loc_reindex_all_miss", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(legacy_get_loc_reindex_checksum(
                        black_box(&source),
                        black_box(&target_range_values),
                    ));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_range_index_indexers);
criterion_main!(benches);
