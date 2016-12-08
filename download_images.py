from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os


def get_soup(url):
    return BeautifulSoup(requests.get(url).text)


def main():
    image_type = "moon_fish"
    query = "moonfish"
    num_per = 28
    for page in xrange(5):
        url = 'http://www.bing.com/images/search?q={query}&web.count={num}&web.offset={off}&qft=+filterui:imagesize-custom_600_600'.format(
            query=query,
            num=num_per,
            off=num_per*page,
        )

        soup = get_soup(url)
        images = [a['src'] for a in soup.find_all("img", {"src": re.compile("mm.bing.net")})]

        for img in images:
            raw_img = urllib2.urlopen(img).read()
            cntr = len([i for i in os.listdir("extra-train-images/LAG/") if image_type in i]) + 1
            f = open('extra-train-images/LAG/{image_type}_{page}_{cntr}.jpg'.format(
                image_type=image_type,
                page=page,
                cntr=str(cntr),
            ), 'wb')
            f.write(raw_img)
            f.close()


if __name__ == "__main__":
    main()