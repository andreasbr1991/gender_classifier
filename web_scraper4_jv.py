import codecs
import ssl
import urllib.request
from urllib.error import URLError, HTTPError
import socket
import http.client
from bs4 import BeautifulSoup
from polyglot.text import Text
import pickle

#Based on http://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text and http://www.nltk.org/book/ch03.html

def filelister(filename):
    f=codecs.open(filename,encoding='iso-8859-1')
    filelist=f.readlines()
    return filelist

            
def webscraper(url_filename,urls_boys_filename,urls_girls_filename,pos_boys_filename,pos_girls_filename,text_boys_filename,text_girls_filename):
    urllist=filelister(url_filename)
    girl_names=sorted(filelister('pigenavne.txt'),key=len,reverse=True)
    boy_names=sorted(filelister('drengenavne.txt'),key=len,reverse=True)
    unisex_names=sorted(filelister('unisexnavne.txt'),key=len,reverse=True) 
        
    boy_names_edt=[boy_name.strip().lower() for boy_name in boy_names if boy_name not in unisex_names]
    girl_names_edt=[girl_name.strip().lower() for girl_name in girl_names if girl_name not in unisex_names] 
    
    urls_boys_file=codecs.open(urls_boys_filename,'w',encoding='utf-8')
    urls_girls_file=codecs.open(urls_girls_filename,'w',encoding='utf-8')
    
    text_boys_file=open(text_boys_filename,'wb')
    text_girls_file=open(text_girls_filename,'wb')
    pos_boys_file=open(pos_boys_filename,'wb')
    pos_girls_file=open(pos_girls_filename,'wb')
    
    text_list_boys=[]
    pos_list_boys=[]
    text_list_girls=[]
    pos_list_girls=[]

    for url in urllist:
        url=url.strip()
        author=None
        author_forename=None
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page,'html5lib')
        try:
            for lines in soup.find_all('div',class_='article_date'): 
                 if lines.find('a', class_='authorName')!=None:
                    print(url)
                    author=lines.find('a', class_='authorName').text
                    author_forename=author.split()[0].strip().lower()
        except (ValueError,IndexError):
            print('ValueError or IndexError')
        if author_forename in boy_names_edt:
            urls_boys_file.write(url.strip()+','+author_forename)
            urls_boys_file.write('\n')
            for article in soup.find_all('div' , class_='remaining_paragraphs'):
                for p in article.find_all('p'):
                    try:
                        visible_text = p.get_text()
                        text=Text(visible_text,hint_language_code='da')
                        text_list_boys.append((author_forename,visible_text))
                        pos=text.pos_tags
                        pos_list_boys.append((author_forename,pos))
                    except ValueError:
                        pass
        elif author_forename in girl_names_edt:
            urls_girls_file.write(url.strip()+','+author_forename)
            urls_girls_file.write('\n')
            for article in soup.find_all('div' , class_='remaining_paragraphs'):
                for p in article.find_all('p'):
                    try:
                        visible_text = p.get_text()
                        text=Text(visible_text,hint_language_code='da')
                        text_list_girls.append((author_forename,visible_text))
                        pos=text.pos_tags
                        pos_list_girls.append((author_forename,pos))
                    except ValueError:
                        pass
                  
    try:
        pickle.dump(text_list_boys,text_boys_file)
        pickle.dump(pos_list_boys,pos_boys_file)
        pickle.dump(text_list_girls,text_girls_file)
        pickle.dump(pos_list_girls,pos_girls_file)
    except ValueError as e:
        pass
    urls_boys_file.close()
    urls_girls_file.close()
    text_boys_file.close()
    text_girls_file.close()
    
                
def main(): 
    webscraper('urls_jv.txt','urls_jv_boys.txt','urls_jv_girls.txt','pos_jv_boys.p','pos_jv_girls.p','text_jv_boys.p','text_jv_girls.p')    
if __name__ == "__main__":
   main()