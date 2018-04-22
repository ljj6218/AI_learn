from sklearn.feature_extraction.text import CountVectorizer
text_content = ["life is short,i like python","life is too long,i dislike python"]
a = CountVectorizer()
b = a.fit_transform(text_content).toarray()
print(b)




