from GoogleNews import GoogleNews
from newspaper import Article
import pandas as pd
from newspaper import Config
import nltk


nltk.download('punkt')
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent

googlenews = GoogleNews(start='01/01/2019',end='01/01/2020')
googlenews.search('Bitcoin')
result=googlenews.result()
df=pd.DataFrame(result)
print(df.head())

for i in range(2,20):
    googlenews.getpage(i)
    result=googlenews.result()
    df=pd.DataFrame(result)

list=[]
for ind in df.index:
    dict={}
    article = Article(df['link'][ind],config=config)
    print(article)
    try:
        article.download()
        article.parse()
        article.nlp()
        dict['Date']=df['date'][ind]
        dict['Media']=df['media'][ind]
        dict['Title']=article.title
        dict['Article']=article.text
        dict['Summary']=article.summary
        list.append(dict)
    except:
        continue
news_df=pd.DataFrame(list)
news_df.to_csv("articles.csv")