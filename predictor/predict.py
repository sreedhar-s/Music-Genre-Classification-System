def predict_gen(meta1):
    import pickle
    import os
    from django.conf import settings
    path = os.path.join(settings.MODELS, 'models.p')
    with open('models.p', 'rb') as f:
        data = pickle.load(f)
    knn = data['knn']
    lgn = data['lgn']
    pred = knn.predict([meta1])
    return lgn[pred[0]]


