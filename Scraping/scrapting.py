import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin, urlparse


def subpages(urls):
    urlPages = []
    seen_urls = set()
    visitedpages=set()
    counter=0

    if isinstance(urls, list):

        for url in urls:
            print("it's just a string")
            suburls = findUrls(url)
            print("You found all the urls")
            visitedpages.add(url)
            for urlpage in suburls:
                counter+=1
                if counter <= 143923:
                    print("strart check the url ",counter)
                    if urlpage.startswith(url) and urlpage not in seen_urls:
                        print("url",counter," pass the check")
                        print("urlpage", urlpage)
                        urlPages.append(urlpage)
                        seen_urls.add(urlpage)
                        print("-----------------------")
                        print()
                        print("start check the next url")
                        print()
                else:
                    return urlPages


    elif isinstance(urls, str):
        
        print("it's just a string")
        suburls = findUrls(urls)
        print("You found all the urls")
        visitedpages.add(urls)
        for urlpage in suburls:
            counter+=1
            if counter <= 143923:
                print("strart check the url ",counter)
                if urlpage.startswith(urls) and urlpage not in seen_urls:
                    print("url",counter," pass the check")
                    print("urlpage", urlpage)
                    urlPages.append(urlpage)
                    seen_urls.add(urlpage)
                    print("-----------------------")
                    print()
                    print("start check the next url")
                    print()
            else:
                return urlPages

    return urlPages

def findUrls(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    suburls = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
    
    return suburls


def scraping(urls):

    counter=0
    # To handle any error in the url
    for url in urls:
        result=[]
        try:
            response = requests.get(url)
            response.raise_for_status()
            print("response finished")
        except requests.RequestException as e:
            print(f"Error fetching URL: {e}")
            
        counter+=1
        print('url' ,counter)
        soup = BeautifulSoup(response.content, 'html.parser')
        try:
            main_content = soup.find('div', class_='padding-bottom-60')
            print("found the main conent finished")
            text= main_content.find_all(['h2','h3','h4','p', 'a', 'span', 'strong'])
            text = [t.text.strip() for t in text]
            print("found the text finished")
            for i in text:
                try:
                    result.append(i)
                    print("finished the appending")
                except:
                    pass
            
            ScraptingDataPath=Path("ScraptingData")
            
            if not ScraptingDataPath.exists():
                ScraptingDataPath.mkdir()

            base_url = "https://u.ae/en/information-and-services"
            base_url1 = "https://u.ae/en/information-and-services/"
            print("strat adding the docs")
            if url == base_url:
                pagename = "information-and-services"
            else:
                pagename = url.replace(base_url1, "").replace("/", "_") 
            
            file_path = ScraptingDataPath / f'{pagename}.txt'
            if not file_path.exists():
                print("new url")
                with open(file_path, 'a') as f:
                    f.write("\n".join(result))
                    print("finished adding the docs")
        except:
            print("couldn't find the main content")

    return print("finished the data scrapping")



# base_url = 'https://u.ae/en/information-and-services'
# urls=subpages(base_url)
# print(urls)
# scraping(urls)
# print("scrapting done!")
# print("finished the first subpages scrapting")
# print()
# print()
# print(urls)

urls=[]
with open("file.txt" , "r") as f:
    urls.extend(line.strip() for line in f)
    print(urls)
    suburls=subpages(urls)
    print(suburls)
    scraping(suburls)
    print("scrapting done!")
    print("finished the first subpages scrapting")
    print()
    print()
    print(suburls)


with open("file.txt" , "r") as f:
    file_contents = f.read()

with open("file.txt" , "a") as f:
    for suburl in suburls:
        if suburl not in file_contents:
           f.write(suburl)
           f.write("\n")

            
