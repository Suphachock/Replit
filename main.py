from flask import Flask, render_template, request, Markup, jsonify
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
import itertools
from gensim.models.tfidfmodel import TfidfModel
import spacy
from spacy import displacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

app = Flask(__name__)
app.config["UPLOAD PATH"] = "upload"


@app.route("/", methods=["GET", "POST"])
def upload_file():
    articles = []
    all_text = []
    tf = []
    bow = []
    gg = []
    find_word = request.form.get("text")
    if request.method == "GET":
        dir = './upload'
        for f in os.listdir(dir):
            os.path.join(dir, f)
            os.remove(os.path.join(dir, f))
    if request.method == "POST" and request.form['action'] == 'Submit':
        dir = './upload'
        for f in os.listdir(dir):
            os.path.join(dir, f)
            os.remove(os.path.join(dir, f))
        for f in request.files.getlist('file_name'):
            test = os.path.join(app.config["UPLOAD PATH"], f.filename)
            f.save(os.path.join(app.config["UPLOAD PATH"], f.filename))
            all_text.append(test)

        for i in all_text:
            # Read TXT file
            f = open(i, "r", encoding="utf8")
            article = f.read()
            # Tokenize the article: tokens
            tokens = word_tokenize(article)
            # Convert the tokens into lowercase: lower_tokens
            lower_tokens = [t.lower() for t in tokens]
            # Retain alphabetic words: alpha_only
            alpha_only = [t for t in lower_tokens if t.isalpha()]
            # Remove all stop words: no_stops
            no_stops = [
                t for t in alpha_only if t not in stopwords.words('english')]
            # Instantiate the WordNetLemmatizer
            wordnet_lemmatizer = WordNetLemmatizer()
            # Lemmatize all tokens into a new list: lemmatized
            lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
            # list_article
            articles.append(lemmatized)
            dictionary = Dictionary(articles)

        def BOW(articles):
            a = []
            corpus = [dictionary.doc2bow(a) for a in articles]
            # Save the second document: doc
            doc = corpus[0]
            # Sort the doc for frequency: bow_doc
            bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)
            for word_id, word_count in bow_doc[:5]:
                (dictionary.get(word_id), word_count)
            total_word_count = defaultdict(int)
            for word_id, word_count in itertools.chain.from_iterable(corpus):
                total_word_count[word_id] += word_count
            # Create a sorted list from the defaultdict: sorted_word_count
            sorted_word_count = sorted(
                total_word_count.items(), key=lambda w: w[1], reverse=True)
            for word_id, word_count in sorted_word_count[:5]:
                final = (dictionary.get(word_id), word_count)
                a.append(final)
            return a

        def tf_idf(articles):
            b = []
            corpus = [dictionary.doc2bow(a) for a in articles]
            # Save the second document: doc
            doc = corpus[0]
            tfidf = TfidfModel(corpus)
            # Calculate the tfidf weights of doc: tfidf_weights
            tfidf_weights = tfidf[doc]
            # Sort the weights from highest to lowest: sorted_tfidf_weights
            sorted_tfidf_weights = sorted(
                tfidf_weights, key=lambda w: w[1], reverse=True)
            # Print the top 5 weighted words
            for term_id, weight in sorted_tfidf_weights[:5]:
                final2 = dictionary.get(term_id), weight
                b.append(final2)
            return b
        bow = BOW(articles)
        tf = tf_idf(articles)
        data = {
            'result1': bow,
            'result2': tf,
        }
        return render_template("find.html", **data)

    if (request.method == "POST" and request.form['action'] == 'Search'):
        find_word = request.form.get("text")
        dir = './upload'
        tmp_file = []
        textfile = []
        for f in os.listdir(dir):
            test = None
            test1 = []
            gg = os.path.join(dir, f)
            rr = os.path.join(f)
            textfile.append(rr)
            f = open(gg, "r", encoding="utf8")
            article = f.read()
            tokens = word_tokenize(article)
            # Convert the tokens into lowercase: lower_tokens
            lower_tokens = [t.lower() for t in tokens]
            # Retain alphabetic words: alpha_only
            alpha_only = [t for t in lower_tokens if t.isalpha()]
            # Remove all stop words: no_stops
            no_stops = [
                t for t in alpha_only if t not in stopwords.words('english')]
            # Instantiate the WordNetLemmatizer
            wordnet_lemmatizer = WordNetLemmatizer()
            # Lemmatize all tokens into a new list: lemmatized
            lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
            # list_article
            articles.append(lemmatized)
            test1.append(lemmatized)
            dictionary = Dictionary(articles)
            test = Dictionary(test1)
            computer_id = test.token2id.get(find_word)
            tmp_file.append(computer_id)

        def BOW(articles):
            a = []
            corpus = [dictionary.doc2bow(a) for a in articles]
            # Save the second document: doc
            doc = corpus[0]
            # Sort the doc for frequency: bow_doc
            bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)
            for word_id, word_count in bow_doc[:5]:
                (dictionary.get(word_id), word_count)
            total_word_count = defaultdict(int)
            for word_id, word_count in itertools.chain.from_iterable(corpus):
                total_word_count[word_id] += word_count
            # Create a sorted list from the defaultdict: sorted_word_count
            sorted_word_count = sorted(
                total_word_count.items(), key=lambda w: w[1], reverse=True)
            for word_id, word_count in sorted_word_count[:5]:
                final = (dictionary.get(word_id), word_count)
                a.append(final)
            return a

        def tf_idf(articles):
            b = []
            corpus = [dictionary.doc2bow(a) for a in articles]
            # Save the second document: doc
            doc = corpus[0]
            tfidf = TfidfModel(corpus)
            # Calculate the tfidf weights of doc: tfidf_weights
            tfidf_weights = tfidf[doc]
            # Sort the weights from highest to lowest: sorted_tfidf_weights
            sorted_tfidf_weights = sorted(
                tfidf_weights, key=lambda w: w[1], reverse=True)
            # Print the top 5 weighted words
            for term_id, weight in sorted_tfidf_weights[:5]:
                final2 = dictionary.get(term_id), weight
                b.append(final2)
            return b

        bow = BOW(articles)
        tf = tf_idf(articles)

        data = {
            'result1': bow,
            'result2': tf,
            'result3': tmp_file,
            'find_word': find_word,
            'fn': textfile
        }
        return render_template("find.html", **data)
    return render_template('index.html')


@app.route('/lab2.html', methods=['GET', 'POST'])
def lab2():
    if request.method == "POST":
        nlp = spacy.load('en_core_web_sm')
        firstname = request.form['firstname'].strip()
        doc = nlp(firstname)
        html = displacy.render(doc, style="ent")
        return jsonify({'output': Markup(html)})
    return render_template('lab2.html')


@app.route('/filter', methods=['GET', 'POST'])
def filter1():
    if request.method == "POST":
        nlp = spacy.load('en_core_web_sm')
        firstname = request.form['text'].strip()
        filter = request.form.getlist('filter[]')
        doc = nlp(firstname)
        ft = {"ents": filter}
        html = displacy.render(doc, style="ent", options=ft)
        return jsonify({'output': Markup(html)})
    return render_template('lab2.html')


@app.route('/fakeNews', methods=['GET', 'POST'])
def fakeNews():
    if request.method == "POST":
        model_path = "FakeNewsV1"
        real_news = request.form['text'].strip()
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        def get_prediction(text, convert_to_label=False):
            # prepare our text into tokenized sequence
            inputs = tokenizer(text, padding=True, truncation=True, max_length=512,
                               return_tensors="pt")
            # perform inference to our model
            outputs = model(**inputs)
            # get output probabilities by doing softmax
            probs = outputs[0].softmax(1)
            # executing argmax function to get the candidate label
            d = {
                0: "reliable",
                1: "fake"
            }
            if convert_to_label:
                return d[int(probs.argmax())]
            else:
                return int(probs.argmax())
        predict = get_prediction(real_news, convert_to_label=True)
        # # read the test set
        # test_df = pd.read_csv("fake_news/test_lite.csv", encoding='latin1')
        # # make a copy of the testing set
        # new_df = test_df.copy()
        # # add a new column that contains the author, title and article content
        # new_df["new_text"] = new_df["author"].astype(
        #     str) + " : " + new_df["title"].astype(str) + " - " + new_df["text"].astype(str)
        # # get the prediction of all the test set
        # new_df["label"] = new_df["new_text"].apply(get_prediction)
        # # make the submission file
        # final_df = new_df[["id", "label"]]
        # final_df.to_csv("submit_final.csv", index=False)
        return jsonify({'output': predict})
    return render_template('fakeNews.html')


if __name__ == "__main__":
    app.run(debug=True)
