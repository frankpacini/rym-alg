import ast
import csv
import math
import random
import requests
import time

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


TIME_TO_WAIT = 10
headers = {'Host':'rateyourmusic.com',
    'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:84.0) Gecko/20100101 Firefox/84.0',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language':'en-US,en;q=0.5', 
    'Accept-Encoding':'gzip, deflate, br',
    'Connection':'keep-alive',
    'Cookie':'__utma=187111339.138898477.1591030842.1610913104.1610932533.208; __utmz=187111339.1610847106.204.11.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __gads=ID=eebc2369ac6234ae:T=1591030842:S=ALNI_MbyJw6pumTGT4N9fphHqv_t0xHpNA; _ga=GA1.2.138898477.1591030842; ulv=%2fwBRf4ZTL3cfF9w7K7JNCUcL9H6n9Br%2fQ3DsnoSfuaXYS8WswpsMi1EZb9kzHtTo1533932568392018; is_logged_in=1; username=bluefrankie55; __stripe_mid=8a58777b-3c8a-4d49-81f2-84e42603a6fb53a579; subscription_update_banner_seen3=true; genre_vote_banner_seen=true; __utmc=187111339; sec_bs=578ca40f052bce1e542bd4b54a6902cc; sec_ts=1610932531; sec_id=766cdceb5c00e5efdaebf1ecc10d2554; __utmb=187111339.1.10.1610932533; __utmt=1',
    'Upgrade-Insecure-Requests':'1',
    'TE':'Trailers'
}

def getLinks():
    with open('rym-links.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for page in range(1, 16):
            print(page)
            url = "https://rateyourmusic.com/charts/top/album,single/all-time/" + str(page) + "/"
            time.sleep(random.random() * 3 + 4)
            r = requests.get(url, headers=headers)
            html = r.text
            soup = BeautifulSoup(html, 'html.parser')
            links = soup.find('div', attrs={'class' : 'chart_results chart_results_ charts_page'}).findAll('a', attrs={'class' : 'release'})
            links = ["https://rateyourmusic.com" + e["href"] for e in links]
            positions = soup.findAll('div', attrs={'class' : 'topcharts_position'})
            positions = [int(e.contents[0]) for e in positions]
            for i in range(len(links)):
                writer.writerow([links[i], positions[i]])

def getAlbum(url):
    r = requests.get(url, headers=headers)
    html = r.text
    soup = BeautifulSoup(html)
    s = str(soup.find('div', attrs={'class':'catalog_stats hide-for-small'}).find('script'))
    i = s.find('[', s.find('data.addRows('))
    j = s.find(')', i)
    arr = ast.literal_eval(s[i:j].replace(' ', '').replace('\n', ''))
    arr = [tup[1] for tup in arr]

    avg_rating = float(soup.find('span', attrs={'class' : 'avg_rating'}).string)
    num_ratings = int(soup.find('span', attrs={'class' : 'num_ratings'}).find('span').string.replace(',', ''))
    arr.insert(0, num_ratings)
    arr.insert(0, avg_rating)
    return arr

def getData(links):
    b = np.ones((1500,12))
    if not pd.read_csv("rym.csv").empty:
        b = np.loadtxt(open("rym.csv", "rb"), delimiter=",")

    with open('rym.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        start = np.nonzero(b[:,0]==1.0)[0][0]
        for i in range(start, 1500):
            print(i)
            arr = []
            try:
                arr = getAlbum(links[i])
            except AttributeError:
                print("AttributeError")
                writer.writerows(b.tolist())
            b[i] = np.array(arr)
            time.sleep(TIME_TO_WAIT)
        writer.writerows(b.tolist())

def normX(X):
    X_0 = ((X[:,0] - np.min(X[:,0]))/math.sqrt(np.var(X[:,0])))[:,None]
    X_1 = ((X[:,1] - np.min(X[:,1]))/math.sqrt(np.var(X[:,1])))[:,None]
    X_dist = X[:,2:] / X[:,1][:,None]
    return np.concatenate((X_0,X_1,X_dist), 1)

def normY(y):
    return (np.mean(y) - y)/math.sqrt(np.var(y))

def getModel(X, y):
    norm_X = normX(X)
    norm_y = normY(y)
    #norm_X = np.hstack((norm_X[:,0:2], norm_X[:,2:] * (np.arange(0, 5, 0.5) + 0.5)))
    #norm_X.sort(axis=0)

    ind = range(norm_X.shape[0])
    test_sample = random.sample(ind, math.ceil(len(ind) / 5.0))
    X_test = norm_X[test_sample,:]
    y_test = norm_y[test_sample]

    ind = list(set(ind) - set(test_sample))
    test_sample = random.sample(ind, math.ceil(len(ind) / 4.0))
    X_cv = norm_X[test_sample,:]
    y_cv = norm_y[test_sample]

    ind = list(set(ind) - set(test_sample))
    X_train = norm_X[ind,:]
    y_train = norm_y[ind]
    """
    N = math.ceil(norm_X.shape[0] / 5.0)
    X_test = norm_X[:n,:]
    y_test = norm_y[:n]
    X_train = norm_X[n:,:]
    y_train = norm_y[n:]
    """
    models = []
    train_errors = []
    cv_errors = []
    units = [30, 45, 60, 75, 90]
    ell = range(2,11)
    for e in range(1):
        inputs = keras.Input(shape=(2,))
        x = inputs
        for _ in range(0,10):
            x = layers.Dense(10, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="rym_model")

        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
        )

        history = model.fit(X_train, y_train, batch_size=X_train.shape[0], epochs=200, validation_split=0.2)
        train_errors.append(history.history['loss'][-1])
        cv_errors.append(model.evaluate(X_cv, y_cv, verbose=2))
        print(model.evaluate(X_test, y_test, verbose=2))
        #plt.show()
        #models.append(model.evaluate(X_cv, y_cv, verbose=2))

        print(model.evaluate(norm_X, norm_y, verbose=2))
        # test(model, X, y)
        print()
        save = input("Save model (Y/N)?: ")
        if save == 'y' or save == 'Y':
            model.save('./model6')
        
    """
    plt.plot(ell, train_errors, label='train')
    plt.plot(ell, cv_errors, label='cv')
    plt.xlabel('layers')
    plt.ylabel('error')
    plt.legend()
    plt.show()
    """
    #print(models)
def getSavedModel(X, y):
    norm_X = normX(X)
    norm_y = normY(y)
    for m in ["model", "model2", "model3"]:
        try:
            model = keras.models.load_model(m)
        except OSError:
            continue
        print(model.evaluate(norm_X, norm_y, verbose=2))
        test(model, X, y)
        print()

def test(model, X, y):
    predictions = (model.predict(normX(X)).flatten("F") * -math.sqrt(np.var(y)) + np.mean(y)).astype("int")
    print(np.abs(predictions - y).max())

    ex_0 = [4.63, 3670, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # paranoid android
    ex_1 = [4.83, 789, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # long season
    ex_2 = [4.75, 247, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # blood promise
    ex_3 = [4.65, 1562, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # ain't it funny
    ex_4 = [4.3, 3569, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # airbag
    test_songs = [ex_0, ex_1, ex_2, ex_3, ex_4]
    for ex in test_songs:
        ex = [(ex[0] - np.min(X[:,0]))/math.sqrt(np.var(X[:,0])),(ex[1] - np.min(X[:,1]))/math.sqrt(np.var(X[:,1]))] + [e / ex[1] for e in ex[2:]] 
        print((model.predict(np.transpose(np.array(ex)[:,None])).flatten("F") * -math.sqrt(np.var(y)) + np.mean(y)).astype("int"))

arr = np.array(list(csv.reader(open("rym-links-rand.csv", "r"), delimiter=",")))
links = arr.astype("str")[:,0]

#getData(links)

X = np.array(list(csv.reader(open("rym.csv", "r"), delimiter=","))).astype("float")[:,:2]
y = arr[:,1].astype("int")
end = np.nonzero(X[:,0]==1.0)[0][0]
X = X[:end]
y = y[:end]

getModel(X, y)
# Test loss: 0.045016393065452576, 294
# Test loss: 0.023925159126520157, 456, 3 hidden layers 64 each
# Test loss: 0.01579330675303936, 831, 3 hidden layers 64 eachn

"""
A = [(0.030279256403446198, 449), (0.030062640085816383, 440), (0.0270612221211195, 417), (0.028206802904605865, 382), (0.02457072213292122, 419), (0.02680259943008423, 394), (0.026309024542570114, 391), (0.024724623188376427, 415), (0.024622788652777672, 408), (0.027418872341513634, 389), (0.031290311366319656, 369)]
B = [(0.03585345670580864, 406), (0.02872663177549839, 429), (0.028526337817311287, 448), (0.027055535465478897, 485), (0.028410471975803375, 425), (0.026964852586388588, 411), (0.025625761598348618, 438), (0.026380380615592003, 441), (0.027778303250670433, 460), (0.023944955319166183, 434), (0.02565450221300125, 475)]
C = [(0.028526682406663895, 455), (0.021691404283046722, 461), (0.020739587023854256, 477), (0.018306797370314598, 434), (0.0186074860394001, 444), (0.018728792667388916, 460), (0.0181251373142004, 437), (0.01774759031832218, 427), (0.017941761761903763, 448), (0.020167993381619453, 448), (0.02285384014248848, 462)]

avgs = []
for i in range(11):
    avgs.append(((A[i][0] + A[i][0] + C[i][0]) / 3.0, (A[i][1] + A[i][1] + C[i][1]) / 3.0))
print(avgs)

5 layers for best loss, 2 for best max rank difference
[
    (0.7728960911432902, 898.0), 
    (0.11308382203181584, 792.6666666666666), 
    (0.03218215393523375, 430.6666666666667), 
    (0.02622527815401554, 475.6666666666667), 
    (0.02490420639514923, 475.3333333333333), 
    (0.024207376564542454, 473.0), 
    (0.028359460333983105, 495.0), 
    (0.026232631256182987, 483.3333333333333), 
    (0.025318540011843044, 487.3333333333333), 
    (0.027414605642358463, 469.3333333333333), 
    (0.026623194416364033, 475.3333333333333)]

~90 units for best loss, ~40 for best max rank difference
[
    (0.02969506507118543, 451.0), 
    (0.02727222815155983, 447.0), 
    (0.024954010422031086, 437.0), 
    (0.02490680105984211, 399.3333333333333), 
    (0.02258297676841418, 427.3333333333333), 
    (0.02411133050918579, 416.0), 
    (0.023581062133113544, 406.3333333333333), 
    (0.022398945565025013, 419.0), 
    (0.022395779689153034, 421.3333333333333), 
    (0.025001912688215572, 408.6666666666667), 
    (0.028478154291709263, 400.0)
]

150 is not enough epochs for 2 layers (64 units each) but fits well for 3+ layers

"""