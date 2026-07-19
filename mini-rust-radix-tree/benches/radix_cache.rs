use std::hint::black_box;
use std::time::{Duration, Instant};

use mini_rust_radix_tree::RadixPrefixCache;

const WARMUP_ITERS: usize = 100;
const BENCH_ITERS: usize = 1_000;

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

fn bench_insert() {
    for &(prefix_len, page_size) in &[(128, 1), (1024, 1), (1024, 16)] {
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
    for &(num_prefixes, prefix_len, page_size) in
        &[(1_024, 128, 1), (1_024, 1024, 1), (1_024, 1024, 16)]
    {
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

fn bench_evict() {
    for &(num_prefixes, prefix_len, page_size, evict_size) in
        &[(1_024, 64, 1, 4096), (1_024, 1024, 16, 16_384)]
    {
        let base_cache = build_linear_cache(num_prefixes, prefix_len, page_size);
        time_case(
            &format!("evict/leaf_lru_prefixes_{num_prefixes}_len_{prefix_len}_page_{page_size}"),
            BENCH_ITERS,
            || {
                let mut cache = base_cache.clone();
                black_box(cache.evict(black_box(evict_size)));
            },
        );
    }
}

fn main() {
    println!("mini-rust-radix-tree benchmark");
    println!("warmup_iters={WARMUP_ITERS}, bench_iters={BENCH_ITERS}\n");

    bench_insert();
    bench_match();
    bench_split();
    bench_lock_unlock();
    bench_evict();
}
