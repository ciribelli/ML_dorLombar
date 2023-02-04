# -*- coding: utf-8 -*-
print('##### comeco em server #####')
import pandas as pd
new_data = pd.read_csv('Dataset_spine_unknown.csv')

import pickle
model = pickle.load(open('model.sav', 'rb'))

predictions = model.predict(new_data)
predictions

new_data['inferencias'] =  predictions
new_data.to_csv('resultados.csv', index = False)