from w3lib.html import remove_tags, remove_tags_with_content
from sqlalchemy import create_engine
from datetime import datetime
import pandas as pd
import random
import scrapy


class ArticleSpider(scrapy.Spider):
    name = "articles"
    user_agent = "../../datasets/data/user_agent.xlsx"
    response_stat = {'success': 0,
                     'failure': 0,
                     'others': 0}

    def get_userlist(df):
        user_agent_df = pd.read_excel(df, header=None, names=['percent', 'useragent', 'system']).dropna()
        user_list = user_agent_df['useragent'].values.tolist()
        return (user_list)

    useragents_list = get_userlist(user_agent)
    HEADERS = {
        "User-Agent": random.choice(useragents_list),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    def __init__(self, name=None, year=0, urlpath=None, *args, **kwargs):
        super(ArticleSpider, self).__init__(*args, **kwargs)
        self.start_time = datetime.now()
        self.year = year
        self.urlpath = urlpath

    def get_urls(self):
        print("----------", self.urlpath)
        df = pd.read_json(self.urlpath, lines=True)
        #         df = df.sample(100).copy()
        url_gen = df.loc[pd.DatetimeIndex(df.date).year == int(self.year)]
        return (url_gen.iterrows())

    def start_requests(self):
        print("----------", self.year)
        print("-----------", self.start_time)
        print(self.response_stat)
        url_gen = self.get_urls()
        for _, row in iter(url_gen):
            request = scrapy.Request(url=row[2], callback=self.parse, headers=self.HEADERS)
            #             request = scrapy.Request(url=row[2], callback=self.parse)
            request.meta['indx'] = row[0]
            request.meta['date'] = row[1]
            yield request

    def parse(self, response):
        if response.status == 200:
            self.response_stat['success'] += 1
        elif response.status == 403:
            self.response_stat['failure'] += 1
        else:
            self.response_stat['others'] += 1

        if self.response_stat['success'] % 5000 == 0:
            print(self.response_stat)
        body = response.css("div.artText").get()
        if not body:
            body = response.css("div._3YYSt.clearfix").get()
        elif not body:
            body = response.css("div.fewcent-408590._1_Akb.clearfix").get()
        elif not body:
            body = response.css("div.Normal").getall()
            body = " ".join(body)

        if body:
            content = remove_tags(remove_tags_with_content(body))
            yield {
                'id': response.meta['indx'],
                'date': response.meta['date'],
                'url': response.url,
                'title': response.url.split("/")[-3],
                'category': response.url.split("/")[-5:-3],
                'article': content
            }
        else:
            yield {
                'id': response.meta['indx'],
                'date': response.meta['date'],
                'url': response.url
            }

    def closed(self, response):
        self.ending_time = datetime.now()
        duration = self.ending_time - self.start_time
        print(self.response_stat)
        print(duration)
