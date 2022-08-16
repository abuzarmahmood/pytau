from tqdm import tqdm
from glob import glob
import os
from time import sleep

parallel_temp_path = '/media/bigdata/projects/pytau/pytau/utils'\
        '/batch_utils/parallel_temp'

def get_count():
    return len(glob(os.path.join(parallel_temp_path,"*.json")))

init_count = get_count()

pbar = tqdm(total=init_count)
last_count = get_count()
while get_count():
    sleep(0.1)
    if get_count() < last_count:
        pbar.update(1)
        last_count = get_count()
pbar.close()
