import numpy as np
import scipy.io as sio
import h5py
from vgg19 import VGG19
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from vgg_predictor import DeepOracle, fev

def baseline_model():
    print('extracting layers:')
    print(layers)

    model = DeepOracle(layers)

    model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[])

    return model

if __name__ == '__main__':
    mat_content = sio.loadmat('../data/02mean_d1.mat')
    activation_model = VGG19(weights='imagenet')
    activations = []
    try:
        f = h5py.File('../data/02activations.hdf5', 'r')
        activations = f['activations'][:]
        f.close()
    except:
        images = [ cv2.resize(cv2.imread('../data/images/%g.jpg'%id),(224,224)).astype(np.float32) for id in tqdm(np.arange(956),desc='loading images') ]
        images = np.array(images)

        train_images = images[train_idxs]
        valid_images = images[valid_idxs]

        activation_fetchers = get_activations(base_model, layers)
        for img in tqdm(images):
            img = np.expand_dims(img, axis=0)
            features = [ feature.predict(img) for feature in activation_fetchers ]
            features = np.concatenate(features, axis=3)
            activations.extend([ features ])

        activations = np.concatenate(activations, axis=0)
        f = h5py.File('../data/02activations.hdf5', 'w')
        f.create_dataset('activations', data=activations)
        f.close()
        pass
    X = activations
    Y = mat_content['activity']
    seed = 7
    np.random.seed(seed)

    base_model = VGG19(weights='imagenet')
    base_model_layers = [ layer.name for layer in base_model.layers[1:-5] ]
    layers = np.array(base_model_layers)[[16, 17, 19]]

    sk_params = dict(
            nb_epoch=100,
            batch_size=32,
            verbose=0,
            layers=layers
            )

    scorer = make_scorer(fev, greater_is_better=True)
    estimator = KerasRegressor(build_fn=baseline_model, **sk_params)
    kf = KFold(n_splits=10,shuffle=True,random_state=seed)

    fev_vec = []
    # for train_index, test_index in kf.split(X):
    #     print('TRAIN:', train_index.shape, ' TEST:', test_index.shape)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = Y[train_index], Y[test_index]
    #     model = baseline_model(layers)
    #     model.fit(X_train, y_train, nb_epoch=10, batch_size=32)
    #     y_pred = model.predict(X_test)
    #     fev_vec.extend([ fev(y_test, y_pred) ])
    #     print(np.array(fev_vec).mean())
    results = cross_val_score(estimator, X, Y, cv=kf, scoring=scorer, verbose=10)
