import argparse
import logging
import warnings

from scrapy.crawler import CrawlerProcess

warnings.filterwarnings('ignore')

import sys
sys.path.append('src/scrapy/article/article/spiders/')
from article_spider import ArticleSpider

# useragents_list = get_userlist(user_agent)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('YEAR', help='starting date from which date to start scraping newsurls')
    parser.add_argument('NEWSURL_PATH', help='starting date from which date to start scraping newsurls')
    parser.add_argument('OUTPUT_PATH', help='end date till which date to do scraping newsurls')
    
    args = parser.parse_args()    
    
    logging.getLogger('scrapy').propagate = False    
    process = CrawlerProcess(settings={
                'FEED_URI': args.OUTPUT_PATH,
                'FORMAT': 'jsonlines',
                })
    process.crawl(ArticleSpider, year=args.YEAR, urlpath=args.NEWSURL_PATH)
    process.start()
