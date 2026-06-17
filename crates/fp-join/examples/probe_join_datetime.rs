use std::collections::BTreeMap;
use fp_columnar::Column; use fp_frame::DataFrame; use fp_index::{Index,IndexLabel};
use fp_join::{merge_dataframes_on, JoinType};
use fp_types::Scalar;
use std::hint::black_box; use std::time::Instant;
fn col_dt(vals:Vec<i64>)->Column{ Column::from_values(vals.into_iter().map(Scalar::Datetime64).collect()).unwrap() }
fn build(n:usize, datetime:bool)->(DataFrame,DataFrame){
  let idx=|m:usize| Index::new((0..m as i64).map(IndexLabel::Int64).collect());
  let lk:Vec<i64>=(0..n as i64).collect();              // ordered unique
  let rk:Vec<i64>=(0..n as i64).map(|i|i*2).collect();
  let (lkc,rkc)= if datetime {(col_dt(lk),col_dt(rk))} else {(Column::from_i64_values(lk),Column::from_i64_values(rk))};
  let mut lm=BTreeMap::new(); lm.insert("key".to_string(),lkc);
  lm.insert("lv".to_string(),Column::from_f64_values((0..n).map(|i|i as f64).collect()));
  let left=DataFrame::new_with_column_order(idx(n),lm,vec!["key".into(),"lv".into()]).unwrap();
  let mut rm=BTreeMap::new(); rm.insert("key".to_string(),rkc);
  rm.insert("rv".to_string(),Column::from_f64_values((0..n).map(|i|i as f64*10.0).collect()));
  let right=DataFrame::new_with_column_order(idx(n),rm,vec!["key".into(),"rv".into()]).unwrap();
  (left,right)
}
fn bench(name:&str,l:&DataFrame,r:&DataFrame,it:usize){
  for _ in 0..3{black_box(merge_dataframes_on(l,r,&["key"],JoinType::Inner).unwrap().columns.len());}
  let st=Instant::now(); let mut k=0usize;
  for _ in 0..it{k^=black_box(merge_dataframes_on(l,r,&["key"],JoinType::Inner).unwrap().columns.len());}
  println!("{name:24}: {:.3} ms/call (k={k})", st.elapsed().as_secs_f64()*1000.0/it as f64);
}
fn main(){
  let n=1_000_000usize;
  let (li,ri)=build(n,false); bench("inner_int64_key",&li,&ri,30);
  let (ld,rd)=build(n,true);  bench("inner_datetime_key",&ld,&rd,30);
}
