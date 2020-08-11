
# import packages
import requests
from bs4 import BeautifulSoup as bs
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# import data from url
iphone_snapdeal=[]
url1 = "https://www.snapdeal.com/product/apple-iphone-5c-16-gb/988871559/reviews?page="
url2 = "&sortBy=RECENCY&vsrc=rcnt#defRevPDP"
# Extracting reviews from snapdeal website
for i in range(1,10):
  ip=[]
  base_url = url1+str(i)+url2
  response = requests.get(base_url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content
  temp = soup.findAll("div",attrs={"class","user-review"})# Extracting the content under specific tags
  for j in range(len(temp)):
    ip.append(temp[j].find("p").text)
  iphone_snapdeal=iphone_snapdeal+ip  # adding the reviews of one page to empty list which in future contains all the reviews

# Removing repeated reviews
iphone_snapdeal = list(set(iphone_snapdeal))
# Writing reviews into text file
with open("ip_snapdeal.txt","w",encoding="utf-8") as snp:
    snp.write(str(iphone_snapdeal))
# Joinining all the reviews into single paragraph
ip_rev_string = " ".join(iphone_snapdeal)
# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ",ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ",ip_rev_string)
# words that contained in reviews
ip_reviews_words = ip_rev_string.split(" ")

# import stopwords, positive, negative words text file & split
# stopwords
with open("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/10. Text Mining - Done/stop.txt","r") as sw:
    stopwords = sw.read().split("\n")
  # stopwords = stopwords
# positive words
with open("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/10. Text Mining - Done/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
# negative words
with open("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/10. Text Mining - Done/negative-words.txt","r",encoding='latin-1') as neg:
  negwords = neg.read().split("\n")

# remove stopwords
ip_reviews_words = [w for w in ip_reviews_words if not w in stopwords]

# Joinining all the reviews into single paragraph
# WordCloud can be performed on the string inputs
# so entire reviews combined into single paragraph
# print word cloud
# without stopwords
ip_rev_string = " ".join(ip_reviews_words)
wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)
plt.xlabel("Word Cloud")
plt.show()
# negative word cloud
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])
wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)
plt.xlabel("Negative Word Cloud")
plt.show()
# Positive word cloud
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)
plt.xlabel("Positive Word Cloud")
plt.show()
















































