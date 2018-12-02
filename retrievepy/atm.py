import retrievepy
from retrievepy.pylogger import get_logger
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

LOGGER = get_logger()


class Atm:
    def __init__(self, tlm):
        """ PRIVATE """
        self.__fetcher = retrievepy.RetrieveEng()
        self.__features = []
        self.__train_data = []
        self.__new_data = []
        self.__test_features = []
        self.__current_window = 0
        self.__clf = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.2, )
        self.__testing = False
        self.__predict_resuts = None
        self.__predict_feature_times = []
        self.__feature_times = []

        """ PUBLIC """
        self.tlm = tlm
        self.start_time = None
        self.end_time = None
        self.data = None
        self.n_points = 100

    def __fetch_data(self, stride=1):
        self.__fetcher.fetch(self.tlm, self.start_time, self.end_time, stride=stride)
        if self.__testing:
            self.__new_data = np.array(self.__fetcher.data)
        else:
            self.__train_data = np.array(self.__fetcher.data)

    def __next_window_data(self):
        start = self.n_points * self.__current_window
        end = start + self.n_points
        self.data = self.__fetcher.data[start:end]
        if self.__testing:
            self.__predict_feature_times.append([start, end])
        else:
            self.__feature_times.append([start, end])
        self.__current_window += 1

    def __calc_features(self):
        # features = [self.__mean(), self.__std(), self.__max(), self.__min()]
        features = [self.__max(), self.__min(), self.__mean, self.__std()]
        if self.__testing:
            self.__test_features.append(features)
        else:
            self.__features.append(features)

    def __reset(self):
        self.__current_window = 0
        self.__fetcher = retrievepy.RetrieveEng()
        self.data = None
        self.__testing = False
    """
    FEATURES
    """
    def __mean(self):
        return np.mean(self.data)

    def __std(self):
        return np.std(self.data)

    def __max(self):
        return np.max(self.data)

    def __min(self):
        return np.min(self.data)

    def __skewness(self):
        return skew(self.data)

    def __kurtosis(self):
        return kurtosis(self.data)

    def __energy(self):
        energy = 0
        for data in self.data:
            energy += data**2

        energy *= 1/len(self.data)
        return energy

    def __normalize(self, features):
        return np.apply_along_axis(self.__normalize_fn, 0, features)

    def __normalize_fn(self, data):
        normed = (data - np.mean(data))/np.std(data)
        return normed

    def __reduce(self):
        pca = PCA(n_components=2)
        pca.fit(np.transpose(self.__features))
        self.__features = pca.components_.T

    def __train(self):
        self.__clf.fit(self.__features)

    def __predict(self):
        self.__predict_results = self.__clf.predict(self.__test_features)
        self.__predict_resuts = np.array(self.__predict_resuts)

    def __preprocess(self):
        if self.__testing:
            features = np.array(self.__test_features)
        else:
            features = np.array(self.__features)
        features = self.__normalize(features)
        # self.__reduce()
        if self.__testing:
            self.__test_features = features
        else:
            self.__features = features

    def __plot_problem_windows(self):
        # create the figure
        fig = plt.figure()

        # open one subplot spanning figure
        ax = fig.add_subplot(111)

        # Configure x-ticks
        ax.grid(True)  # turn on the grid

        problem_indicies = self.__predict_feature_times[self.__predict_results == -1]

        times = np.array([])

        for index in problem_indicies:
            tmptimes = np.arange(index[0], index[1])
            times = np.append(times, tmptimes)

        full_time = np.arange(0, len(self.__new_data))

        plt.plot(full_time, self.__new_data,
                 color='gray',
                 marker='.',
                 markerfacecolor='green',
                 markeredgecolor='green',
                 markersize=5)

        plt.plot(times, self.__new_data[times.astype(int)],
                 ls='None',
                 marker='.',
                 markerfacecolor='red',
                 markeredgecolor='red',
                 markersize=7)

        # fig.autofmt_xdate(rotation=50)  # rotate the tick labels

        ax.set_title(self.tlm)  # sets the title of the plot

        fig.tight_layout()  # makes it so labels/ticks are not cut off

        # show the plot
        plt.show()

    def __plot_contours(self):
        plot_range = [-10, 10]
        xx, yy = np.meshgrid(np.linspace(-50, 50, 500), np.linspace(-50, 50, 500))
        # plot the line, the points, and the nearest vectors to the plane
        thingy = np.c_[xx.ravel(), yy.ravel()]
        Z = self.__clf.decision_function(thingy)
        Z = Z.reshape(xx.shape)

        plt.title("Novelty Detection")
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.BuPu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

        s = 40

        b1 = plt.scatter(self.__features[:, 0], self.__features[:, 1], c='white', s=s, edgecolors='k')
        if self.__testing:
            b2 = plt.scatter(self.__test_features[:, 0][self.__predict_results == 1],
                             self.__test_features[:, 1][self.__predict_results == 1],
                             c='blueviolet',
                             s=s,
                             edgecolors='k')
            b3 = plt.scatter(self.__test_features[:, 0][self.__predict_results == -1],
                             self.__test_features[:, 1][self.__predict_results == -1],
                             c='red',
                             s=s,
                             edgecolors='k')
        plt.axis('tight')
        plt.xlim((plot_range[0], plot_range[1]))
        plt.ylim((plot_range[0], plot_range[1]))
        if self.__testing:
            plt.legend([a.collections[0], b1, b2, b3],
                       ["learned frontier", "training observations",
                        "new regular observations", "new abnormal observations"],
                       loc="upper left",
                       prop=matplotlib.font_manager.FontProperties(size=11))
        else:
            plt.legend([a.collections[0], b1],
                       ["learned frontier", "training observations",
                        "new regular observations", "new abnormal observations"],
                       loc="upper left",
                       prop=matplotlib.font_manager.FontProperties(size=11))
        plt.show()

    def predict(self, st, et, plot=True, scatter=True):
        self.__reset()
        self.__testing = True
        self.start_time = st
        self.end_time = et
        self.__fetch_data()
        while self.__current_window * self.n_points + self.n_points < len(self.__fetcher.data):
            self.__next_window_data()
            self.__calc_features()
        self.__predict_feature_times = np.array(self.__predict_feature_times)
        self.__preprocess()
        self.__predict()
        if scatter:
            self.__plot_contours()
        if plot:
            self.__plot_problem_windows()

    def train(self, st, et, plot=True):
        self.__reset()
        self.start_time = st
        self.end_time = et
        self.__fetch_data(stride=1)
        while self.__current_window * self.n_points + self.n_points < len(self.__fetcher.data):
            self.__next_window_data()
            self.__calc_features()
        self.__preprocess()
        self.__train()
        if plot:
            self.__plot_contours()

