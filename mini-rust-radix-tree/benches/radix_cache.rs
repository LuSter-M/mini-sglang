use std::hint::black_box;
use std::thread;
use std::time::{Duration, Instant};

use mini_rust_radix_tree::{ConcurrentRadixPrefixCache, RadixPrefixCache};

const WARMUP_ITERS: usize = 100;
const BENCH_ITERS: usize = 10_000;
const FAST_BENCH_ITERS: usize = 1_000_000;
const EVICT_BENCH_ITERS: usize = 1_000;
const PARALLEL_THREADS: usize = 8;
const PARALLEL_LOOKUP_ITERS_PER_THREAD: usize = 100_000;
const PARALLEL_MATCH_ITERS_PER_THREAD: usize = 20_000;

fn sequential_tokens(start: i32, len: usize) -> Vec<i32> {
    (0..len).map(|i| start + i as i32).collect()
}

fn cache_indices(start: i32, len: usize) -> Vec<i32> {
    (0..len).map(|i| start + i as i32).collect()
}

fn build_linear_cache(
    num_prefixes: usize,
    prefix_len: usize,
    page_size: usize,
) -> RadixPrefixCache {
    let mut cache = RadixPrefixCache::new(page_size);
    for prefix_id in 0..num_prefixes {
        let base = (prefix_id * prefix_len) as i32;
        let input_ids = sequential_tokens(base, prefix_len);
        let indices = cache_indices(base * 10, prefix_len);
        cache.insert_prefix(&input_ids, &indices);
    }
    cache
}

fn build_shared_prefix_cache(
    num_branches: usize,
    shared_len: usize,
    branch_len: usize,
    page_size: usize,
) -> RadixPrefixCache {
    let mut cache = RadixPrefixCache::new(page_size);
    for branch_id in 0..num_branches {
        let mut input_ids = sequential_tokens(0, shared_len);
        input_ids.extend(sequential_tokens(
            10_000 + (branch_id * branch_len) as i32,
            branch_len,
        ));

        let mut indices = cache_indices(0, shared_len);
        indices.extend(cache_indices(
            100_000 + (branch_id * branch_len) as i32,
            branch_len,
        ));

        cache.insert_prefix(&input_ids, &indices);
    }
    cache
}

fn time_case<F>(name: &str, iters: usize, mut f: F)
where
    F: FnMut(),
{
    for _ in 0..WARMUP_ITERS {
        f();
    }

    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    print_result(name, iters, elapsed);
}

fn print_result(name: &str, iters: usize, elapsed: Duration) {
    let total_ns = elapsed.as_nanos();
    let avg_ns = total_ns as f64 / iters as f64;
    let ops_per_sec = 1_000_000_000_f64 / avg_ns;
    println!(
        "{name:<54} total={:>10.3} ms  avg={:>10.1} ns/op  throughput={:>12.1} ops/s",
        elapsed.as_secs_f64() * 1_000.0,
        avg_ns,
        ops_per_sec,
    );
}

fn time_parallel_case<F>(name: &str, threads: usize, iters_per_thread: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    thread::scope(|scope| {
        for worker_id in 0..threads {
            let f = &f;
            scope.spawn(move || {
                for iter in 0..WARMUP_ITERS {
                    f(worker_id, iter);
                }
            });
        }
    });

    let start = Instant::now();
    thread::scope(|scope| {
        for worker_id in 0..threads {
            let f = &f;
            scope.spawn(move || {
                for iter in 0..iters_per_thread {
                    f(worker_id, iter);
                }
            });
        }
    });
    let elapsed = start.elapsed();
    print_result(name, threads * iters_per_thread, elapsed);
}

fn time_parallel_case_with_state<S, I, F>(
    name: &str,
    threads: usize,
    iters_per_thread: usize,
    init: I,
    f: F,
) where
    S: Send,
    I: Fn() -> S + Sync,
    F: Fn(usize, usize, &mut S) + Sync,
{
    thread::scope(|scope| {
        for worker_id in 0..threads {
            let init = &init;
            let f = &f;
            scope.spawn(move || {
                let mut state = init();
                for iter in 0..WARMUP_ITERS {
                    f(worker_id, iter, &mut state);
                }
            });
        }
    });

    let start = Instant::now();
    thread::scope(|scope| {
        for worker_id in 0..threads {
            let init = &init;
            let f = &f;
            scope.spawn(move || {
                let mut state = init();
                for iter in 0..iters_per_thread {
                    f(worker_id, iter, &mut state);
                }
            });
        }
    });
    let elapsed = start.elapsed();
    print_result(name, threads * iters_per_thread, elapsed);
}

fn bench_insert() {
    for &(prefix_len, page_size) in &[(128, 1), (1024, 1), (1024, 16), (4096, 16)] {
        let input_ids = sequential_tokens(0, prefix_len);
        let indices = cache_indices(0, prefix_len);
        time_case(
            &format!("insert_prefix/cold_len_{prefix_len}_page_{page_size}"),
            BENCH_ITERS,
            || {
                let mut cache = RadixPrefixCache::new(page_size);
                black_box(cache.insert_prefix(black_box(&input_ids), black_box(&indices)));
            },
        );
    }
}

fn bench_match() {
    for &(num_prefixes, prefix_len, page_size) in &[
        (1_024, 128, 1),
        (1_024, 1024, 1),
        (1_024, 1024, 16),
        (4_096, 1024, 16),
        (8_192, 512, 16),
    ] {
        let mut cache = build_linear_cache(num_prefixes, prefix_len, page_size);
        let query = sequential_tokens(((num_prefixes / 2) * prefix_len) as i32, prefix_len + 32);
        time_case(
            &format!("match_prefix/prefixes_{num_prefixes}_len_{prefix_len}_page_{page_size}"),
            BENCH_ITERS * 10,
            || {
                black_box(cache.match_prefix(black_box(&query)));
            },
        );
    }
}

fn bench_lookup() {
    for &(num_prefixes, prefix_len, page_size) in &[
        (1_024, 128, 1),
        (1_024, 1024, 1),
        (1_024, 1024, 16),
        (4_096, 1024, 16),
        (8_192, 512, 16),
    ] {
        let cache = build_linear_cache(num_prefixes, prefix_len, page_size);
        let query = sequential_tokens(((num_prefixes / 2) * prefix_len) as i32, prefix_len + 32);
        time_case(
            &format!("lookup_indices/prefixes_{num_prefixes}_len_{prefix_len}_page_{page_size}"),
            BENCH_ITERS * 10,
            || {
                black_box(cache.lookup_indices(black_box(&query)));
            },
        );

        let mut output = Vec::with_capacity(prefix_len);
        time_case(
            &format!(
                "lookup_indices_into/prefixes_{num_prefixes}_len_{prefix_len}_page_{page_size}"
            ),
            BENCH_ITERS * 10,
            || {
                black_box(cache.lookup_indices_into(black_box(&query), black_box(&mut output)));
                black_box(&output);
            },
        );
    }
}

fn bench_split() {
    let original = sequential_tokens(0, 1024);
    let original_indices = cache_indices(0, 1024);
    let mut divergent = sequential_tokens(0, 512);
    divergent.extend(sequential_tokens(50_000, 512));
    let divergent_indices = cache_indices(100_000, 1024);

    time_case(
        "split_on_partial_match/mid_node_len_1024_page_1",
        BENCH_ITERS,
        || {
            let mut cache = RadixPrefixCache::new(1);
            cache.insert_prefix(&original, &original_indices);
            black_box(cache.insert_prefix(black_box(&divergent), black_box(&divergent_indices)));
        },
    );
}

fn bench_lock_unlock() {
    let mut cache = build_shared_prefix_cache(256, 128, 32, 1);
    let query = sequential_tokens(0, 128);
    let handle = cache.match_prefix(&query).handle;

    time_case("lock_unlock_handle/shared_path", BENCH_ITERS * 10, || {
        cache.lock_handle(black_box(handle));
        cache.unlock_handle(black_box(handle));
        black_box(cache.size_info());
    });
}

fn bench_concurrent_wrapper() {
    let mut raw_cache = build_linear_cache(1_024, 128, 1);
    let wrapped_cache = ConcurrentRadixPrefixCache::from_cache(raw_cache.clone());
    let query = sequential_tokens(512 * 128, 160);

    time_case("mutex_wrapper/match_prefix_raw", BENCH_ITERS * 10, || {
        black_box(raw_cache.match_prefix(black_box(&query)));
    });

    time_case(
        "rwlock_wrapper/match_prefix_write_locked",
        FAST_BENCH_ITERS,
        || {
            black_box(wrapped_cache.match_prefix(black_box(&query)));
        },
    );

    time_case(
        "rwlock_wrapper/lookup_indices_raw",
        FAST_BENCH_ITERS,
        || {
            black_box(raw_cache.lookup_indices(black_box(&query)));
        },
    );

    time_case(
        "rwlock_wrapper/lookup_indices_read_locked",
        FAST_BENCH_ITERS,
        || {
            black_box(wrapped_cache.lookup_indices(black_box(&query)));
        },
    );

    let mut raw_lookup_output = Vec::with_capacity(128);
    time_case(
        "rwlock_wrapper/lookup_indices_into_raw",
        FAST_BENCH_ITERS,
        || {
            black_box(
                raw_cache.lookup_indices_into(black_box(&query), black_box(&mut raw_lookup_output)),
            );
            black_box(&raw_lookup_output);
        },
    );

    let mut locked_lookup_output = Vec::with_capacity(128);
    time_case(
        "rwlock_wrapper/lookup_indices_into_read_locked",
        FAST_BENCH_ITERS,
        || {
            black_box(
                wrapped_cache
                    .lookup_indices_into(black_box(&query), black_box(&mut locked_lookup_output)),
            );
            black_box(&locked_lookup_output);
        },
    );

    let cache = ConcurrentRadixPrefixCache::new(1);
    let input_ids = sequential_tokens(0, 128);
    let indices = cache_indices(0, 128);
    time_case(
        "rwlock_wrapper/reset_insert_two_write_locks",
        FAST_BENCH_ITERS,
        || {
            cache.reset();
            black_box(cache.insert_prefix(black_box(&input_ids), black_box(&indices)));
        },
    );

    time_case(
        "rwlock_wrapper/reset_insert_one_write_lock",
        FAST_BENCH_ITERS,
        || {
            cache.with_write_cache(|cache| {
                cache.reset();
                black_box(cache.insert_prefix(black_box(&input_ids), black_box(&indices)));
            });
        },
    );

    let parallel_cache = ConcurrentRadixPrefixCache::from_cache(build_linear_cache(4_096, 128, 1));
    let parallel_queries: Vec<_> = (0..4_096)
        .map(|prefix_id| sequential_tokens(prefix_id * 128, 160))
        .collect();
    time_parallel_case(
        &format!("rwlock_wrapper/parallel_lookup_indices_{PARALLEL_THREADS}_threads"),
        PARALLEL_THREADS,
        PARALLEL_LOOKUP_ITERS_PER_THREAD,
        |worker_id, iter| {
            let prefix_id = (worker_id * 257 + iter) % 4_096;
            let query = &parallel_queries[prefix_id];
            black_box(parallel_cache.lookup_indices(black_box(&query)));
        },
    );

    time_parallel_case_with_state(
        &format!("rwlock_wrapper/parallel_lookup_indices_into_{PARALLEL_THREADS}_threads"),
        PARALLEL_THREADS,
        PARALLEL_LOOKUP_ITERS_PER_THREAD,
        || Vec::with_capacity(128),
        |worker_id, iter, output| {
            let prefix_id = (worker_id * 257 + iter) % 4_096;
            let query = &parallel_queries[prefix_id];
            black_box(parallel_cache.lookup_indices_into(black_box(&query), black_box(output)));
            black_box(&output);
        },
    );

    time_parallel_case(
        &format!("rwlock_wrapper/parallel_match_prefix_{PARALLEL_THREADS}_threads"),
        PARALLEL_THREADS,
        PARALLEL_MATCH_ITERS_PER_THREAD,
        |worker_id, iter| {
            let prefix_id = (worker_id * 257 + iter) % 4_096;
            let query = &parallel_queries[prefix_id];
            black_box(parallel_cache.match_prefix(black_box(&query)));
        },
    );
}

fn bench_evict() {
    for &(num_prefixes, prefix_len, page_size, evict_size) in
        &[(1_024, 64, 1, 4096), (1_024, 1024, 16, 16_384)]
    {
        let base_cache = build_linear_cache(num_prefixes, prefix_len, page_size);
        time_case(
            &format!("evict/leaf_lru_prefixes_{num_prefixes}_len_{prefix_len}_page_{page_size}"),
            EVICT_BENCH_ITERS,
            || {
                let mut cache = base_cache.clone();
                black_box(cache.evict(black_box(evict_size)));
            },
        );
    }
}

fn main() {
    println!("mini-rust-radix-tree benchmark");
    println!(
        "warmup_iters={WARMUP_ITERS}, bench_iters={BENCH_ITERS}, fast_bench_iters={FAST_BENCH_ITERS}, evict_bench_iters={EVICT_BENCH_ITERS}\n"
    );

    bench_insert();
    bench_match();
    bench_lookup();
    bench_split();
    bench_lock_unlock();
    bench_concurrent_wrapper();
    bench_evict();
}
