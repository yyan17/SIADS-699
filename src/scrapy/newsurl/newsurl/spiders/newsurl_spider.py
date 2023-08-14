from datetime import datetime
from itertools import count

import scrapy
from scrapy.linkextractors import LinkExtractor
from urllib3.exceptions import HTTPError


class NewsURLSpider(scrapy.Spider):
    name = "newsurl"
    le1 = LinkExtractor()

    base_url = "https://economictimes.indiatimes.com/"
    valid_url1 = 'articleshow'
    valid_url2 = base_url
    
    def __init__(self, pred_gen, *args, **kwargs):
        super(NewsURLSpider, self).__init__(*args, **kwargs)
#         self.output_path = output_path
        self.pred_gen = pred_gen

        self.start_time = datetime.now()
        self.iter_count = count(0)    
        
    def start_requests(self):
        for indx, param in iter(self.pred_gen): 
            date, predicate = param
            url = self.base_url + predicate + '.cms'
#             print(url)
            request = scrapy.Request(url=url, callback=self.parse,
                                                errback=self.errback_httpbin)
            request.meta['date'] = date
            request.meta['iter_count'] = self.iter_count
            yield request
            
    def parse(self, response):
        num = 0
        for link in self.le1.extract_links(response):             
            if self.valid_url1 in link.url and self.valid_url2 in link.url:
                num += 1
                item = {
                        'id': next(response.meta['iter_count']),
                        'date': response.meta['date'],
                        'url': link.url
                            }
                yield item
#                 self.data.append(item)
                
# error handing adopted from below stackoverflow webpage                
# https://stackoverflow.com/questions/31146046/how-do-i-catch-errors-with-scrapy-so-i-can-do-something-when-i-get-user-timeout               
    def errback_httpbin(self, failure):
        # log all errback failures,
        # in case you want to do something special for some errors,
        # you may need the failure's type
        self.logger.error(repr(failure))

        #if isinstance(failure.value, HttpError):
        if failure.check(HTTPError):
            # you can get the response
            response = failure.value.response
            self.logger.error('HttpError on %s', response.url)

#         #elif isinstance(failure.value, DNSLookupError):
#         elif failure.check(DNSLookupError):
#             # this is the original request
#             request = failure.request
#             self.logger.error('DNSLookupError on %s', request.url)

#         #elif isinstance(failure.value, TimeoutError):
#         elif failure.check(TimeoutError):
#             request = failure.request
#             self.logger.error('TimeoutError on %s', request.url)        
    
    def closed(self, response):
        self.ending_time = datetime.now()
        duration = self.ending_time - self.start_time
        print(duration)        
       