import gensim
import pandas as pd

df = pd.read_json("Cell_Phones_and_Accessories_5.json", lines=True)
reviewed_text = df.reviewText.apply(gensim.utils.simple_preprocess)

model= gensim.models.FastText(window=10, min_count=2, sg=1)

model.build_vocab(reviewed_text, progress_per=1000)

model.train(reviewed_text, total_examples=model.corpus_count, epochs= 10)

#TESTING THE MODEL

print(model.wv.most_similar("bad")) #test the model by displaying most similar words to word bad

print(model.wv.similarity(w1="bad", w2="good")) #print the similarity score between two words
