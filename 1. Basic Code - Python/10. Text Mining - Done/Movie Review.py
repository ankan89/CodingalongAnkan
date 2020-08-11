# import packages
import requests
from bs4 import BeautifulSoup as bs
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import FreqDist

# import data from url
movie_re = []
url1 = "https://www.imdb.com/title/tt4154796/reviews?ref_=tt_urv"
resp = requests.get(url1)
movie1 = bs(resp.text, 'html.parser')
movie1
movie_d = movie1.find_all('div', {'class': 'content'})
movie_d
movie_d1 = movie1.find_all('div', {'class': 'text show-more__control'})
movie_d1
movie_d2 = movie1.find_all('div', {'class': 'review-container'})
movie_d2

# print movie review
len(movie_d1)
movie_d1[24].text
movier = []
for i in range(len(movie_d1)):
    movier.append(movie_d1[i].text)
movier_rev_string = " ".join(movier)

# Removing unwanted symbols incase if exists
movier_rev_string = re.sub("[^A-Za-z" "]+", " ", movier_rev_string).lower()
movier_rev_string = re.sub("[0-9" "]+", " ", movier_rev_string)

# words that contained in  reviews
movier_reviews_words = movier_rev_string.split(" ")

# import stopwords text file
with open("/Users/Ankan/PycharmProjects/Basiccode/1 SC/Text mining/stop.txt", "r") as sw:
    stopwords = sw.read()

# import negative words text file
with open("/Users/Ankan/PycharmProjects/Basiccode/1 SC/Text mining/negative-words.txt", "r", encoding='latin1') as sw:
    negative = sw.read()

# import positive words text file
with open("/Users/Ankan/PycharmProjects/Basiccode/1 SC/Text mining/positive-words.txt", "r") as sw:
    positive = sw.read()

# split stopwords, positive & negative words
stopwords = stopwords.split("\n")
positive = positive.split("\n")
negative = negative.split("\n")

# Joinining all the reviews into single paragraph
movier_rev_string = " ".join(movier_reviews_words)

# WordCloud can be performed on the string inputs. That is the reason we have combined
# entire reviews into single paragraph
# make word cloud
# stopwords
movier_reviews_words = [w for w in movier_reviews_words if not w in stopwords]
wordcloud_movier = WordCloud(
    background_color='black',
    width=1800,
    height=1400
).generate(movier_rev_string)

plt.imshow(wordcloud_movier)
# negative words
movier_neg_in_neg = " ".join([w for w in movier_reviews_words if w in negative])
wordcloud_neg_in_neg = WordCloud(
    background_color='black',
    width=1800,
    height=1400
).generate(movier_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)
# positive words
movier_pos_in_pos = " ".join([w for w in movier_reviews_words if w in positive])
wordcloud_pos_in_pos = WordCloud(
    background_color='black',
    width=1800,
    height=1400
).generate(movier_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)

# plot pie charrt of every review through function
# pie chart shows % of positive & negative words present in one review.
# in pie chart if % of positive words > negative words it means it's positive review.
# if % of positive words < negative words it means it's negative review.

for i in range(len(movier)):
    moviere = re.sub("[^A-Za-z" "]+", " ", movier[i]).lower()
    moviere = re.sub("[0-9" "]+", " ", moviere)
    moviere = moviere.split(" ")

    moviestop = [i for i in moviere if i not in stopwords]
    movieposit = [i for i in moviere if i in positive]
    movienegat = [i for i in moviere if i in negative]

    all_words_frequency = FreqDist(moviere)

    print(all_words_frequency)
    print(all_words_frequency.most_common(10))
    label = ['positive', 'negative']
    count = [len(movieposit), len(movienegat)]
    figureObject, axesObject = plt.subplots()

    # Draw the pie chart
    axesObject.pie(count, labels=label, autopct='%1.2f', startangle=90)
    plt.title(i)
    # Aspect ratio - equal means pie is a circle
    axesObject.axis('equal')
    plt.show()



















