find ./parallel_temp -iname "*.json" | xargs -n1 realpath > ./parallel_temp/parallel_jobs.txt 
cat ./parallel_temp/parallel_jobs.txt |
parallel -k -j 12 --ungroup --noswap --load 100% --progress --memfree 4G python fit_from_file.py {}
