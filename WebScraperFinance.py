

import urllib.request
from bs4 import BeautifulSoup


def scrape_get_tickers(site):
    
    response = urllib.request.urlopen(site)
    website = response.read()    
    soup = BeautifulSoup(website,'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    all_info_dict=[]    
    pure_tickers = []
    for row in table.findAll('tr'):
        col = row.findAll('td')
        if len(col) > 0:
            all_info = dict()
            sector = str(col[3].string.strip())
            industry = str(col[4].string.strip())
            ticker = str(col[0].string.strip())
            for tag1, tag2 in zip(col[0].findAll('a', href=True),col[2].findAll('a', href=True)):               
               all_info['attributes']=tag1['href'],col[1].string.strip(),tag2['href']
            all_info['stock']=ticker   
            all_info['sector']=sector  
            all_info['industry']=industry  
            all_info_dict.append(all_info)           
            pure_tickers.append(ticker)           
            
    return (pure_tickers,all_info_dict)



# to do http://biz.yahoo.com/p/ scraper start with industry, then crawl over each link and save all categories and final ticker name and country
