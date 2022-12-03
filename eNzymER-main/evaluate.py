import eval4ner.muc as muc

import pprint

# ground truth
grount_truths = [
    [('PER', 'John Jones'), ('PER', 'Peter Peters'), ('LOC', 'York')],
    [('PER', 'John Jones'), ('PER', 'Peter Peters'), ('LOC', 'York')],
    [('PER', 'John Jones'), ('PER', 'Peter Peters'), ('LOC', 'York')]
]
# NER model prediction
predictions = [
    [('PER', 'John Jones and Peter Peters came to York')],
    [('LOC', 'John Jones'), ('PER', 'Peters'), ('LOC', 'York')],
    [('PER', 'John Jones'), ('PER', 'Peter Peters'), ('LOC', 'York')]
]
# input texts
texts = [
    'John Jones and Peter Peters came to York',
    'John Jones and Peter Peters came to York',
    'John Jones and Peter Peters came to York'
]
muc.evaluate_all(predictions, grount_truths * 1, texts, verbose=True)