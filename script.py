import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

class NearestNeighborsModel:
    def __init__(self, data, n_neighbors=10):
        self.data = data
        self.n_neighbors = n_neighbors
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        self.encoded_data = None
        self.columns = self.get_columns()

    def get_columns(self):
        return [col.replace(' ', '_').rstrip('_') for col in self.data[0].keys()]

    def fit(self):
        data_matrix = [[row[col] for col in self.columns] for row in self.data]
        self.encoded_data = self.encoder.fit_transform(data_matrix).toarray()
        self.nn.fit(self.encoded_data)

    def find_nearest_neighbors(self, input_data):
        input_data = [input_data.get(col, 'None') for col in self.columns]
        encoded_input_data = self.encoder.transform([input_data]).toarray()
        indices = self.nn.kneighbors(encoded_input_data)
        nearest_neighbors = [self.data[index] for index in indices[0]]
        return nearest_neighbors

    def add_data(self, new_data):
        self.data.extend(new_data)
        self.fit()


model = NearestNeighborsModel(data)
model.fit()

input_data = {
    'Подразделение_1': "Центральный офис",
    'Имя': 'Александр',
    'Фамилия': 'Смирнов'
}

nearest_neighbors = model.find_nearest_neighbors(input_data)
print(nearest_neighbors)