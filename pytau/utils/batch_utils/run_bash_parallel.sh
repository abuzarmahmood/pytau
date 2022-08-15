find ./parallel_temp -iname "*.json" | xargs -n1 realpath > ./parallel_temp/parallel_jobs.txt 
cat ./parallel_temp/parallel_jobs.txt |
parallel -k -j 28 --ungroup --noswap --load 100% --progress --memfree 4G bash single_process.sh {}
