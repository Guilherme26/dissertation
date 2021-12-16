import scrapy
import pandas as pd

from time import sleep
from scrapy.selector import Selector
from selenium import webdriver


class TEDSubtitles(scrapy.Spider):
    name = "TEDSubtitles"
    df = pd.read_csv("../data/TED_Talks.csv")
    start_urls = df.link.values

    def __init__(self):
        self.driver = webdriver.Chrome(executable_path="./chromedriver")
        self.driver.maximize_window()

    def parse(self, response):
        self.driver.get(response.url)
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        sleep(3)

        page_body = Selector(text=self.driver.page_source)

        subtitles = []
        for item in page_body.xpath('//div[contains(@class, " Grid__cell flx-s:1 p-r:4 ")]'):
            subtitles.append(" ".join(item.xpath('//a[contains(@class, "t-d:n hover/bg:gray-l.5")]/text()').getall()))
        
        return {"url": response.url, "subtitle": " ".join(subtitles)}
