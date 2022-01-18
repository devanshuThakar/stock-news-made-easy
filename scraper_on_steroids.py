import pandas as pd
from nltk import FreqDist
from collections import Counter
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
import numpy as np
import praw
import gensim
from gensim.models import Word2Vec
from scipy import spatial
import emoji
from heapq import heappop, heappush, heapify
from tqdm import tqdm

punctuation = punctuation + "0987654321”"


def content_text(text):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    useless_words = Counter()
    might_be_useful = Counter()
    visited = {}

    for word in text:
        # update count off all words in the line that are in stopwords
        word = emoji.demojize(word)

        if word in stopwords or visited.get(word, False):
            useless_words.update([word])
        else:
            if any(punct in word for punct in punctuation):
                visited[word] = True
                useless_words.update([word])
                continue

           # update count off all words in the line that are not in stopwords or have punctuations
            might_be_useful.update([word])

    return set(might_be_useful)

# Make sure to download glove model to your system, and enter its location in File variable
def load_glove_model(File=r"C:\Location_of_glove\glove.6B.50d.txt"):
    print("Loading Glove Model")
    glove_model = {}
    with open(File, 'r', encoding="utf8") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def copy_post(submission):
    post_dict = {}
    post_dict["title"] = submission.title
    post_dict["score"] = submission.score
    post_dict["id"] = submission.id
    post_dict["url"] = submission.url
    post_dict["comms_num"] = submission.num_comments
    post_dict["created"] = submission.created
    post_dict["body"] = submission.selftext
    return post_dict


def scrape(number_of_posts=60, keyword='trade'):

    subreddit = reddit.subreddit('wallstreetbets')   # Chosing the subreddit
    query = [keyword]
    posts = {}

    for i in range(30):
        print(f"Scraping posts: round {i+1}")

        common_string = ""

        print("Bringing posts for :", "|".join(query))
        for item in query:
            # new hi rakhna

            for submission in subreddit.search(item, sort="new", limit=number_of_posts):
                common_string += submission.title + " " + submission.selftext + " "

                posts[submission.id] = copy_post(submission)

        print("Processing Posts")
        copy_common_string = common_string.replace("\n", " ")

        data = []
        tokenized_data = []

        # iterate through each sentence in the file
        for sentence in sent_tokenize(copy_common_string):
            temp = []

            # tokenize the sentence into words
            for word in word_tokenize(sentence):
                temp.append(word.lower())
                tokenized_data.append(word.lower())

            data.append(temp)

        # Create CBOW model
        print("Creating CBOW model")
        model1 = gensim.models.Word2Vec(data)
        temp_lst = content_text(tokenized_data)

        best_words = []

        print("Matching Similarity")
        for word in tqdm(temp_lst):
            flag = 0
            maxi = 0

            if flag == 0:
                try:
                    simil = 1 - \
                        spatial.distance.cosine(
                            model1.wv[word], model1.wv[keyword])

                    best_words.append((word, simil))

                except KeyError:
                    pass
                    # print(word)

        best_words = sorted(best_words, key=lambda x: x[1], reverse=True)
        best_words = list(map(lambda x: x[0], best_words))
        [best_words.remove(word) for word in query]

        query.extend(best_words[:2])
        print(f"Got {len(posts)} unique posts")

    return posts


if __name__ == "__main__":
    print("Scrapping Reddit")

    clint_id = "4i6QIlNm5-Xq8R********"
    clinet_screct = "uijggGBEuri4Tos**********"
    user_agent = "my user agent"
    username = "********************"
    password = "********************"

    reddit = praw.Reddit(client_id=clint_id,  # my client id
                         client_secret=clinet_screct,  # your client secret
                         user_agent=user_agent,  # user agent name
                         username=username,     # your reddit username
                         password=password)     # your reddit password
    posts = scrape()

    data = posts.values()
    data = pd.DataFrame(data)
    data.to_csv("new_posts_scrape_1.csv")