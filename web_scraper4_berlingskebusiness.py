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
    links=[]
    
    for url in urllist:
        
        url=url.strip()
        try:
            page=urllib.request.urlopen(url)
            try:
                soup = BeautifulSoup(page,'html5lib')
                [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
                
                try:
                    if 'berlingske.dk' in url:
                        for header in soup.find_all('header'):
                            for h2 in header.find_all('h2',class_="post-title"):
                                link=h2.find('a').get('href')
                                if type(link)==str:
                                    if link not in links and 'facebook' not in link.lower() and 'twitter' not in link.lower() and 'instagram' not in link.lower() and 'google' not in link.lower()  and 'pinterest' not in link.lower() and 'snapchat' not in link.lower():
                                        links.append(link)
                    elif 'business.dk' in url:
                        for header in soup.find_all('header'):
                            for h2 in header.find_all('h2',class_="entry-title"):
                                link=h2.find('a').get('href')
                                if type(link)==str:
                                    if link not in links and 'facebook' not in link.lower() and 'twitter' not in link.lower() and 'instagram' not in link.lower() and 'google' not in link.lower()  and 'pinterest' not in link.lower() and 'snapchat' not in link.lower():
                                        links.append(link)
                except ValueError:
                    pass
            except TypeError:
                pass  
        except HTTPError as e:
           print('Error message:',e.msg)
        except URLError as e:
            print('Error reason:',e.reason)
        except http.client.IncompleteRead as e:
            page = e.partial
        except socket.gaierror:
            pass
        except TimeoutError:
            pass
    links_checker_boys=set()
    links_checker_girls=set()
    if len(links)>0:        
        for link in links:
            author=None
            author_forename=None
            try:
                page=urllib.request.urlopen(link)
                try:
                    soup = BeautifulSoup(page,'html5lib')
                    try:
                        if soup.find(class_='author')!=None:
                            author=soup.find(class_='author').text
                            author_forename=author.split()[0].strip().lower()
                        elif soup.find(rel='author')!=None:
                            author=soup.find(rel='author').text
                            author_forename=author.split()[0].strip().lower()
                    except (ValueError,IndexError):
                        print('ValueError or IndexError')
                    
                    for content in soup.find_all('div', class_="entry-content"):
                        for p in content.find_all('p'):
                            try:
                                if author_forename in boy_names_edt and link not in links_checker_boys:
                                    print(link)
                                    links_checker_boys.add(link)
                                    urls_boys_file.write(link.strip()+','+author_forename+'\n')
                                    visible_text = p.get_text() 
                                    text=Text(visible_text,hint_language_code='da')
                                    text_list_boys.append((author_forename,visible_text))
                                    pos=text.pos_tags
                                    pos_list_boys.append((author_forename,pos))
                                elif author_forename in girl_names_edt and link not in links_checker_girls:
                                    print(link)
                                    links_checker_girls.add(link)
                                    urls_girls_file.write(link.strip()+','+author_forename+'\n')
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
            except URLError as e:
                print('Error reason:',e.reason)
            except http.client.IncompleteRead as e:
                page = e.partial
            except socket.gaierror:
                pass
            except TimeoutError:
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
    pos_boys_file.close()
    pos_girls_file.close()
    text_boys_file.close()
    text_girls_file.close()
    
if __name__ == "__main__":
    webscraper('urls_berlingske.txt','urls_berlingske_boys.txt','urls_berlingske_girls.txt','pos_berlingske_boys.p','pos_berlingske_girls.p','text_berlingske_boys.p','text_berlingske_girls.p')
    webscraper('urls_business.txt','urls_business_boys.txt','urls_business_girls.txt','pos_business_boys.p','pos_business_girls.p','text_business_boys.p','text_business_girls.p')