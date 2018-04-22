import sklearn
from sklearn.feature_extraction import DictVectorizer
instances = [{'city': '北京','temperature':100},
             {'city': '上海','temperature':60},
             {'city': '深圳','temperature':30}]
a = DictVectorizer()
x = a.fit_transform(instances).toarray()
# y = a.fit_transform(instances).()
print(a.inverse_transform(x))
print(x)