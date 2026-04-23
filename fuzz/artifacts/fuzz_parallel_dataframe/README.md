# fuzz_parallel_dataframe artifacts

New crashes from local or CI fuzz runs land here first.
Minimize them with cargo fuzz tmin fuzz_parallel_dataframe <artifact> and then promote the minimized input into fuzz/corpus/fuzz_parallel_dataframe/.

Run under TSan (separate campaign, not shareable with ASan) via:
  RUSTFLAGS="-Zsanitizer=thread" cargo +nightly fuzz run fuzz_parallel_dataframe
