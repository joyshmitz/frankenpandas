# fuzz_sql_read artifacts

New crashes from local or CI fuzz runs land here first.
Minimize them with cargo fuzz tmin fuzz_sql_read <artifact> and then promote the minimized input into fuzz/corpus/fuzz_sql_read/.
