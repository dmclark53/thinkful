# constants.py
from tensorflow.keras.metrics import TruePositives
from tensorflow.keras.metrics import FalsePositives
from tensorflow.keras.metrics import TrueNegatives
from tensorflow.keras.metrics import FalseNegatives
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import AUC

API_KEY = 'iTR_lY8voYedlA_gJm7PqY4ej3nit0EtyVzzef7qX6FWh_glj5CISjc165B99fLtDPBXbO_3b6U8xzoN9r_dmgO8naulMIZ1RH9OVWSVaxUMH1kKD5YIVJ4FL10GXnYx'
SEARCH_URL = 'https://api.yelp.com/v3/businesses/search'
BUSINESSES_URL = 'https://api.yelp.com/v3/businesses'

HEADERS = {'Authorization': f'Bearer {API_KEY}'}

SEARCH_PARAMS = {'term': 'restaurants',
                 'location': 'San Jose, CA',
                 'limit': 50,
                 'offset': 0}

BUSINESSES_DICT = {
    'id': [],
    'name': [],
    'is_closed': [],
    'review_count': [],
    'rating': [],
    'distance': []
}

REVIEWS_DICT = {
    'id': [],
    'rating': [],
    'text': [],
    'time_created': [],
    'url': []
}

FULL_REVIEWS_DICT = {
    'rating': [],
    'text': []
}

'''
Modeling
'''

# VOCAB_SIZE = 10386
VOCAB_SIZE = 9566

METRICS = [
    TruePositives(name='tp'),
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn'),
    BinaryAccuracy(name='accuracy'),
    Precision(name='precision'),
    Recall(name='recall'),
    AUC(name='auc')
]

PARAMS = {
    'batch_size': '',
    'epochs': ''
}
