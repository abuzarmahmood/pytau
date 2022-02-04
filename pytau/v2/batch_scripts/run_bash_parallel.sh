cat parallel_jobs.txt |
parallel -k -j 12 --ungroup --noswap --load 100% --progress --memfree 4G python fit_from_file.py {}
