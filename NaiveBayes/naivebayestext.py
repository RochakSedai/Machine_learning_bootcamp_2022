from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

training_data =  fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)



# print('\n'.join(training_data.data[0].split('\n')[:10]))
# print('Target is: ', training_data.target_names[training_data.target[0]])

# we just count the word occurences
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)
# print(count_vector.vocabulary_)

# we transform the word occurences into tfidf
# tfidfVectorizer = CountVectorizer + TfidfTransformer
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

model = MultinomialNB().fit(x_train_tfidf, training_data.target)

new = ['This has nothing to do with church or religion', 'Software engineering is getting hotter and hotter nowadays']

x_new_counts = count_vector.transform(new)
x_new_tfidf = tfidf_transformer.transform(x_new_counts)

predicted = model.predict(x_new_tfidf)
print(predicted)

for doc, category in zip(new, predicted):
    print('%r -----------> %s' %(doc, training_data.target_names[category]))