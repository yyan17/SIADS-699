#!/bash
echo "hi"
scrapy crawl articles -o ../../../datasets/articles_2004.jl -a year=2004 --nolog
scrapy crawl articles -o ../../../datasets/articles_2005.jl -a year=2005 --nolog
scrapy crawl articles -o ../../../datasets/articles_2006.jl -a year=2006 --nolog
scrapy crawl articles -o ../../../datasets/articles_2007.jl -a year=2007 --nolog
scrapy crawl articles -o ../../../datasets/articles_2008.jl -a year=2008 --nolog
scrapy crawl articles -o ../../../datasets/articles_2009.jl -a year=2009 --nolog
scrapy crawl articles -O ../../../datasets/articles_2010.jl -a year=2010 --nolog
scrapy crawl articles -o ../../../datasets/articles_2011.jl -a year=2011 --nolog
scrapy crawl articles -O ../../../datasets/articles_2012.jl -a year=2012 --nolog
scrapy crawl articles -O ../../../datasets/articles_2013.jl -a year=2013 --nolog
scrapy crawl articles -O ../../../datasets/articles_2014.jl -a year=2014 --nolog
scrapy crawl articles -O ../../../datasets/articles_2015.jl -a year=2015 --nolog
scrapy crawl articles -O ../../../datasets/articles_2016.jl -a year=2016 --nolog
scrapy crawl articles -O ../../../datasets/articles_2017.jl -a year=2017 --nolog
scrapy crawl articles -O ../../../datasets/articles_2018.jl -a year=2018 --nolog
scrapy crawl articles -O ../../../datasets/articles_2019.jl -a year=2019 --nolog
scrapy crawl articles -O ../../../datasets/articles_2020.jl -a year=2020 --nolog
scrapy crawl articles -O ../../../datasets/articles_2021.jl -a year=2021 --nolog