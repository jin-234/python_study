from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

(train_x,train_y),(test_x,test_y)=imdb.load_data(num_words=50)#빈도수 상위 500개의 단어만 사용함

from tensorflow.keras.preprocessing.sequence import pad_sequences
test_seq=pad_sequences(test_x,maxlen=100)

# print(test_seq.shape)#(25000, 100)
# print(test_seq[0])
# [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  1  2  2 14 31  6  2 10 10  2  2  5  4  2  7  4
#   2  2  2  2  4  2  9  2  2  2 10 10 13  2  2  2  2  2  2 28  2 14 31 23
#  27  2 29  2  2  8  2 14  2  2  8  2 46  5 27  2 16  2  2 38 32 25  2  2
#   2 14  6  2]

model=load_model('/home/jin/deeplearning_prj/20260617/imbd_model.keras')

# print(model.predict(test_seq[0:1]))
# print(test_y[0])

word_to_index=imdb.get_word_index()
# for key,value in word_to_index.items():
#     if value==1:
#         print('key,value:',key,value)
# print(word_to_index['this'])


positive_review_str='I really enjoyed this film. The visuals were beautiful, and the music matched the atmosphere perfectly. The characters were interesting, and the story kept me engaged throughout the entire movie.'
negative_review_str='This was one of the worst movies I have ever watched. I went into it with fairly high expectations because the trailer looked promising and the cast seemed talented, but the actual film was a complete disappointment from start to finish. The story was poorly written, filled with plot holes, and lacked any sense of direction. Many scenes felt unnecessary and dragged on for far too long, making the movie feel much longer than it actually was.'
import re
def new_sentence_tokenization(sentence_arg):
    new_sentence=re.sub(r'[^a-zA-Z\s]',' ',sentence_arg).lower()
    encoded=[]
    for word in new_sentence.split(' '):
        try:
            if word_to_index[word]<=500:
                encoded.append(word_to_index[word]+3)
            else:
                encoded.append(2)
        except KeyError:
            encoded.append(2)
            
    pad_new=pad_sequences([encoded],maxlen=100)
    print(pad_new)
    score=float(model.predict(pad_new))
    print('score:',score)
    if score>0.5:
        print('{:.2f}확률로 긍정 리뷰'.format(score*100))
    else:
        print('{:.2f}확률로 부정 리뷰'.format((1-score)*100))

new_sentence_tokenization(positive_review_str)
new_sentence_tokenization(negative_review_str)
