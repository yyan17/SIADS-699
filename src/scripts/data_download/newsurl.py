import argparse
import logging
import warnings
from datetime import date, datetime

from scrapy.crawler import CrawlerProcess

warnings.filterwarnings('ignore')

import sys
sys.path.append('../')
sys.path.append('../scrapy/newsurl/newsurl/spiders/')
from newsurl_spider import NewsURLSpider
from utils import scraputil


beg_date = date(2008, 1, 1)
end_date = date(2023, 5, 31)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('BEG_DATE', help='starting date from which date to start scraping newsurls')
    parser.add_argument('END_DATE', help='end date till which date to do scraping newsurls')
    parser.add_argument('OUTPUT_PATH', help='end date till which date to do scraping newsurls')
    
    args = parser.parse_args()    

    beg_date = datetime.strptime(args.BEG_DATE, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.END_DATE, '%Y-%m-%d').date()
    
    print(beg_date, end_date)
    # create predicates generator which is used by the scrapy class NewURLSpider class to iterate over requests    
    pred_gen = enumerate(scraputil.create_predicates(beg_date, end_date))

#     output_path = args.OUTPUT_PATH
    logging.getLogger('scrapy').propagate = False
    process = CrawlerProcess(settings={
                'FEED_URI': args.OUTPUT_PATH,
                'FORMAT': 'jsonlines',
                })
    
    process.crawl(NewsURLSpider, pred_gen=pred_gen)
    process.start()
