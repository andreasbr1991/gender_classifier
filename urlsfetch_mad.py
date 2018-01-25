#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import urllib.request
from bs4 import BeautifulSoup
import codecs
import sys

def urlsfetch(url,filename):
    f=codecs.open(filename,'w',encoding='utf-8')
           
    # Based on http://stackoverflow.com/questions/1080411/retrieve-links-from-web-page-using-python-and-beautifulsoup
    
    with urllib.request.urlopen(url) as page:
        soup = BeautifulSoup(page,"html5lib")
        for div in soup.find_all('div',attrs={'class':'span4 center '}):
            print(div)
            l=div.find('img')['src'] 
            l_split=str(l).split('src=')
            if l_split[1].startswith('http'):
                if '.dk' in l_split[1]:
                    l1_split=l_split[1].split('.dk')
                    url_out=l1_split[0].replace('%3A%2F%2F','://')
                    f.write(url_out+'.dk')
                    f.write('\n')
               
               


 
def main():
    url=sys.argv[1]
    filename=sys.argv[2]
    urlsfetch(url,filename)

if __name__ == "__main__":
    main()

