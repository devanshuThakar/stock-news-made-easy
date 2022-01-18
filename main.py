from fastapi import FastAPI, Body, Request, File, UploadFile, Form
import uvicorn
# import torch
# from rouge import Rouge
# import pandas as pd
# import numpy as np
import re
from transformers import BartForConditionalGeneration, BartTokenizer, PegasusForConditionalGeneration, PegasusTokenizer
from pathlib import Path
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from functools import lru_cache
import praw

templates = Jinja2Templates(directory="html")
scrapped_posts=[]
class Input(BaseModel):
    input: str

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def prediction_pegasuscnn(text,path_peg,tokenizer,model):

    # path_peg = Path("./augmented_pegasus_cnn")
    # # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # tokenizer = PegasusTokenizer.from_pretrained(path_peg)
    # model = PegasusForConditionalGeneration.from_pretrained(path_peg)#.to(device)
    tweet = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%.\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%\+.~#?&//=]*)', '', text, flags=re.MULTILINE) # to remove links that start with HTTP/HTTPS in the tweet
    tweet = re.sub(r'http\S+', '', tweet,flags=re.MULTILINE)
    tweet = re.sub(r'[-a-zA-Z0–9@:%.\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%\+.~#?&//=]*)', '', tweet, flags=re.MULTILINE) # to remove other url links

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    emoji_pattern.sub(r'', tweet)
    # tweet = ' '.join(re.sub('/[\u{1F600}-\u{1F6FF}]/'," ",tweet).split()) # for emojis

    tweet = re.sub(r"#(\w+)", ' ', tweet, flags=re.MULTILINE)
    tweet = re.sub(r"@(\w+)", ' ', tweet, flags=re.MULTILINE)
    text = tweet
    src_text = [text]
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAA",len(src_text))
    tab = []
    for i in range(len(src_text)):
        inputs = tokenizer(src_text[i], max_length=1024, return_tensors='pt')
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=1000, early_stopping=True)
        tab.append([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
    return tab[0][0]

def prediction_bartcnn(text,path_bart,tokenizer,model):

    # path_bart = Path("./bart_cnn_final")
    # # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # tokenizer = BartTokenizer.from_pretrained(path_bart)
    # model = BartForConditionalGeneration.from_pretrained(path_bart)#.to(device)
    tweet = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%.\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%\+.~#?&//=]*)', '', text, flags=re.MULTILINE) # to remove links that start with HTTP/HTTPS in the tweet
    tweet = re.sub(r'http\S+', '', tweet,flags=re.MULTILINE)
    tweet = re.sub(r'[-a-zA-Z0–9@:%.\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%\+.~#?&//=]*)', '', tweet, flags=re.MULTILINE) # to remove other url links

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    emoji_pattern.sub(r'', tweet)
    # tweet = ' '.join(re.sub('/[\u{1F600}-\u{1F6FF}]/'," ",tweet).split()) # for emojis

    tweet = re.sub(r"#(\w+)", ' ', tweet, flags=re.MULTILINE)
    tweet = re.sub(r"@(\w+)", ' ', tweet, flags=re.MULTILINE)
    text = tweet
    src_text = [text]
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAA",len(src_text))
    tab = []
    for i in range(len(src_text)):
        inputs = tokenizer(src_text[i], max_length=1024, return_tensors='pt')
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=1000, early_stopping=True)
        tab.append([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
    return tab[0][0]

def scrape(number_of_posts=1,keyword='stocks',
        clint_id="WWgBYx3DIJp2Y*********",clinet_screct="SzcENicLrwcA0jky-R5_sDaS*******",
        user_agent="my user agent",username = "*****nlp_21",password = "******"):
    reddit = praw.Reddit(client_id=clint_id,#my client id
                     client_secret=clinet_screct,  #your client secret
                     user_agent=user_agent, #user agent name
                     username = username,     # your reddit username
                     password = password)     # your reddit password
    print("ASDSAFADFA")
    subreddit = reddit.subreddit('wallstreetbets')   # Chosing the subreddit

    query = [keyword]
    post_list=[]
    print(query)
    for item in query:
        print("ASDSAFADFA")
        for submission in subreddit.search(item,sort = "new", limit=number_of_posts): # new hi rakhna  
            print("ASDSAFADFA")
            post_dict={}
            post_dict["title"]=submission.title
            post_dict["score"]=submission.score
            post_dict["id"]=submission.id
            post_dict["url"]=submission.url
            post_dict["comms_num"]=submission.num_comments
            post_dict["created"]=submission.created
            post_dict["body"]=submission.selftext
            post_list.append(post_dict)

    return post_list

def assign_scrapped_posts(posts):
    scrapped_posts=posts

@app.get("/bart/", response_class=HTMLResponse)
def bart_summary(request: Request):
    bart_summary=[]
    path_bart = Path("./bart_cnn_final")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BartTokenizer.from_pretrained(path_bart)
    model = BartForConditionalGeneration.from_pretrained(path_bart)#.to(device)
    for post in scrapped_posts:
        x=post['body']
        if(len(list(x.split()))>40):
            summary = prediction_bartcnn(post['body'],path_bart,tokenizer,model)
        elif(len(x)==0):
            summary = "No text to summarize."
        else:
            summary = x
        bart_summary.append(summary)
    return templates.TemplateResponse("summarized_posts.html",context={"request": request, "model":'bart',"summary_list":bart_summary})

@app.get("/pegasus/", response_class=HTMLResponse)
def pegasus_summary(request: Request):
    pegasus_summary=[]
    path_peg = Path("./augmented_pegasus_cnn")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(path_peg)
    model = PegasusForConditionalGeneration.from_pretrained(path_peg)#.to(device)
    for post in scrapped_posts:
        x=post['body']
        if(len(list(x.split()))>40):
            summary = prediction_pegasuscnn(post['body'],path_peg,tokenizer,model)
        elif(len(x)==0):
            summary = "No text to summarize."
        else:
            summary = x
        pegasus_summary.append(summary)
    return templates.TemplateResponse("summarized_posts.html",context={"request": request, "model":'pegasus',"summary_list":pegasus_summary})

@app.get("/")
def handleform(request: Request):
    return templates.TemplateResponse("test.html",context={"request": request})

@app.post("/", response_class=HTMLResponse)
@lru_cache(100)
def handleform(request: Request, keyword: str = Form(...), numberOfPosts: str = Form(...)):
    global scrapped_posts
    print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
    scrapped_posts=scrape(keyword=keyword,number_of_posts=int(numberOfPosts))
    print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
    # assign_scrapped_posts(scrapped_posts)
    # x='http://'+host+':'+port+'/'+'stonks/'
    pegasus="pegasus/"
    bart = "bart/"
    return templates.TemplateResponse("scrapped_posts.html",context={"request":request,"keyword":keyword,"post_list":scrapped_posts, "bart":bart, "pegasus":pegasus})



@app.get("/debug")
def handleform(request: Request):
    summary1 = "Give an input"
    summary2 = "Give an input"
    return templates.TemplateResponse("predict.html",context={"request": request,"summary1": summary1, "summary2":summary2})


@app.post("/debug")
@lru_cache(100)
def handleform(request: Request, input: str = Form(...)):
    summary1 = prediction_bartcnn(input)
    summary2 = prediction_bartcnn(input)
    return templates.TemplateResponse("predict.html",context={"request": request,"summary1": summary1, "summary2":summary2})

if __name__=="__main__":
    port="8888"
    host="127.0.0.1"
    # host = "0.0.0.0"
    uvicorn.run(app,host=host,port=int(port))
