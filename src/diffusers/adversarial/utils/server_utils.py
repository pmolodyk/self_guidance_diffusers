from urllib import response
import requests
import time
from os import path


def check_free(stalked, numbers):
    # stalked = ['img87', 'img84'], numbers = [0, 1]
    assert len(stalked) == len(numbers)
    url = "http://101.200.38.10/e628de4c-eeb4-413e-aaad-b39f9583f72d/hulabcluster/index.html#detail"
    while True:
        response = requests.get(url)
        all_html = response.content.decode()
        for i, st in enumerate(stalked):
            needed = all_html.split('"detail-tab"')[2].split(st + '      ')[1]
            n = numbers[i]
            # print(st + '     ', all_html.split('"detail-tab"')[2].split(st + '      ')[1][:300])
            if needed.split('[')[1 + n].split('|')[-1] == '<br>':
                return
        time.sleep(10)
