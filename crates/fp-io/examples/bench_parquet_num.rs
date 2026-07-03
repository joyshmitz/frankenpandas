use std::path::Path;
fn main(){
    let mut best=u128::MAX;
    for _ in 0..6 {
        let t=std::time::Instant::now();
        let df=fp_io::read_parquet(Path::new("/tmp/bench_num.parquet")).unwrap();
        let mut acc=0usize; for name in df.column_names(){acc+=df.column(name.as_str()).unwrap().values().len();}
        std::hint::black_box(acc); best=best.min(t.elapsed().as_nanos());
    }
    println!("fp read_parquet 1Mx6 numeric: {:.2}ms", best as f64/1e6);
}
