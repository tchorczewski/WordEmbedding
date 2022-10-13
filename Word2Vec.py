import gensim
import pandas as pd

df = pd.read_json("Cell_Phones_and_Accessories_5.json", lines=True)

reviewed_text = df.reviewText.apply(gensim.utils.simple_preprocess) #simple preprocessing for sake of implementation

model = gensim.models.Word2Vec(
    window=10, #size of window around target word
    min_count= 2,# minimal number of words in a sentence needed for it to be considered a training data
    workers= 4, # number of threads that will work (training is using CPU not GPU)
    sg=1
)

model.build_vocab(reviewed_text, progress_per=1000) #progress_per =1000 indicates after how many processed words the program will show progress

print(model.epochs) #default number of loops over the whole dataset

model.train(reviewed_text, total_examples=model.corpus_count, epochs= 10) #training the model Bag of words method

#TESTING THE MODEL

print(model.wv.most_similar("bad")) #test the model by displaying most similar words to word bad

print(model.wv.similarity(w1="bad", w2="good")) #print the similarity score between two words

