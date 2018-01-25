import codecs
import urllib.request
from urllib.error import URLError, HTTPError
import http.client
import socket
from bs4 import BeautifulSoup
from polyglot.text import Text
import pickle

#Based on http://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text and http://www.nltk.org/book/ch03.html

def filelister(filename):
    f=codecs.open(filename,encoding='iso-8859-1')
    filelist=f.readlines()
    return filelist


def webscraper(url_boys,url_girls,pos_boys_filename,pos_girls_filename,text_boys_filename,text_girls_filename):
    boys_urllist=filelister(url_boys)
    girls_urllist=filelister(url_girls)
    girl_names=sorted(filelister('pigenavne.txt'),key=len,reverse=True)
    boy_names=sorted(filelister('drengenavne.txt'),key=len,reverse=True)
    unisex_names=sorted(filelister('unisexnavne.txt'),key=len,reverse=True) 
        
    boy_names_edt=[boy_name.strip().lower() for boy_name in boy_names if boy_name not in unisex_names]
    girl_names_edt=[girl_name.strip().lower() for girl_name in girl_names if girl_name not in unisex_names] 
    
    
    text_boys_file=open(text_boys_filename,'wb')
    text_girls_file=open(text_girls_filename,'wb')
    pos_boys_file=open(pos_boys_filename,'wb')
    pos_girls_file=open(pos_girls_filename,'wb')

    text_list_boys=[]
    pos_list_boys=[]
    text_list_girls=[]
    pos_list_girls=[]
    
    links_boys=set()
    links_girls=set()
    for url_boyname in boys_urllist:
        url_boyname_split=url_boyname.split(',')
        link=url_boyname_split[0]
        if not link.startswith('https'): 
            try:
                page=urllib.request.urlopen(link)
                try:
                    soup = BeautifulSoup(page,'html5lib')
                    [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
                    try:
                        if soup.find(class_='author')!=None:
                            author=soup.find(class_='author').text
                            author_forename=author.split()[0].strip().lower()
                        elif soup.find(rel='author')!=None:
                            author=soup.find(rel='author').text
                            author_forename=author.split()[0].strip().lower()
                    except (ValueError,IndexError):
                        print('ValueError or IndexError')
                    for article in soup.find_all('article'):
                        for p in article.find_all('p'):
                            try:
                                if author_forename in boy_names_edt  and link not in links_boys:
                                    print(link)
                                    links_boys.add(link)
                                    visible_text = p.get_text() 
                                    text=Text(visible_text,hint_language_code='da')
                                    text_list_boys.append((author_forename,visible_text))
                                    pos=text.pos_tags
                                    pos_list_boys.append((author_forename,pos))
                                elif author_forename in girl_names_edt  and link not in links_boys:
                                    print(link)
                                    links_boys.add(link)
                                    visible_text = p.get_text()
                                    text=Text(visible_text,hint_language_code='da')
                                    text_list_girls.append((author_forename,visible_text))
                                    pos=text.pos_tags
                                    pos_list_girls.append((author_forename,pos))
                            except ValueError:
                                pass
                except TypeError:
                    pass
            except HTTPError as e:
               print('Error message:',e.msg)
               continue
            except URLError as e:
                print('Error reason:',e.reason)
            except http.client.IncompleteRead as e:
                page = e.partial
            except socket.gaierror:
                pass
            except TimeoutError:
                pass
            except ValueError:
                pass
            
    for url_girlname in girls_urllist:
        url_girlname_split=url_girlname.split(',')
        link=url_girlname_split[0]
        if not link.startswith('https'): 
            try:
                page=urllib.request.urlopen(link)
                try:
                    soup = BeautifulSoup(page,'html5lib')
                    [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
                    try:
                        if soup.find(class_='author')!=None:
                            author=soup.find(class_='author').text
                            author_forename=author.split()[0].strip().lower()
                        elif soup.find(rel='author')!=None:
                            author=soup.find(rel='author').text
                            author_forename=author.split()[0].strip().lower()
                    except (ValueError,IndexError):
                        print('ValueError or IndexError')
                    for article in soup.find_all('article'):
                        for p in article.find_all('p'):
                            try:
                                if author_forename in boy_names_edt and link not in links_girls:
                                    print(link)
                                    links_girls.add(link)
                                    visible_text = p.get_text() 
                                    text=Text(visible_text,hint_language_code='da')
                                    text_list_boys.append((author_forename,visible_text))
                                    pos=text.pos_tags
                                    pos_list_boys.append((author_forename,pos))
                                elif author_forename in girl_names_edt and link not in links_girls:
                                    print(link)
                                    links_girls.add(link)
                                    visible_text = p.get_text()
                                    text=Text(visible_text,hint_language_code='da')
                                    text_list_girls.append((author_forename,visible_text))
                                    pos=text.pos_tags
                                    pos_list_girls.append((author_forename,pos))
                            except ValueError:
                                pass
                except TypeError:
                    pass
            except HTTPError as e:
               print('Error message:',e.msg)
               continue
            except URLError as e:
                print('Error reason:',e.reason)
            except http.client.IncompleteRead as e:
                page = e.partial
            except socket.gaierror:
                pass
            except TimeoutError:
                pass
            except ValueError:
                pass        
        

    try:
        pickle.dump(text_list_boys,text_boys_file)
        pickle.dump(pos_list_boys,pos_boys_file)
        pickle.dump(text_list_girls,text_girls_file)
        pickle.dump(pos_list_girls,pos_girls_file)
    except ValueError as e:
        pass

    pos_boys_file.close()
    pos_girls_file.close()
    text_boys_file.close()
    text_girls_file.close()
    
if __name__ == "__main__":
    webscraper('urls_foodblogs_boys.txt','urls_foodblogs_girls.txt','pos_foodblogs_boys.p','pos_foodblogs_girls.p','text_foodblogs_boys.p','text_foodblogs_girls.p')
    webscraper('urls_travelblogs_boys.txt','urls_travelblogs_girls.txt','pos_travelblogs_boys.p','pos_travelblogs_girls.p','text_travelblogs_boys.p','text_travelblogs_girls.p')
   
#    webscraper('urls_berlingske.txt','urls_berlingske_boys.txt','urls_berlingske_girls.txt','pos_berlingske_boys.p','pos_berlingske_girls.p','text_berlingske_boys.p','text_berlingske_girls.p')
#    webscraper('urls_business.txt','urls_business_boys.txt','urls_business_girls.txt','pos_business_boys.p','pos_business_girls.p','text_business_boys.p','text_business_girls.p')
   