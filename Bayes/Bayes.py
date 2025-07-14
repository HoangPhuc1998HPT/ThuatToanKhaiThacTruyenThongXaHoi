from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
from sklearn.preprocessing import LabelEncoder



class Bayes:
    """
    A class to represent a Bayesian model.
    """
    data = 'data_bayes.csv'
    # Đọc dữ liệu
    df = pd.read_csv("data_bayes.csv", encoding='utf-8')

    # Hiển thị thông tin dữ liệu
    df.info()
    df.isnull().sum()  # kiểm tra dữ liệu thiếu

    # Hiển thị kích thước tập dữ liệu
    #df.shape  # (rows, columns)
    # Hiển thị 5 dòng đầu tiên của dữ liệu
    #df.head()  # hiển thị 5 dòng đầu tiên

    # Danh sách tên các cột trong DataFrame
    columns_name = df.columns.tolist()  # Lấy tên các cột trong DataFrame
    print(columns_name)

    # Làm sạch dữ liệu
    df = df.dropna()  # Loại bỏ các dòng có giá trị NaN
    df = df.drop_duplicates()  # Loại bỏ các dòng trùng lặp
    df = df.reset_index(drop=True)  # Đặt lại chỉ mục của DataFrame
    # Hiện thị lại kích thước tập dữ liệu
    print("Kích thước tập dữ liệu sau khi làm sạch:", df.shape)

    # Tìm lóp Ci có xác xuất hậu nghiệm cao nhất


    def value_of_maximum_posterior(self, col_name):
        # Tìm giá trị có xác suất hậu nghiệm cao nhất trong cột col_name
        labels = self.df[col_name].tolist()  # Lấy danh sách các nhãn trong cột col_name
        priors = Counter(labels)

        # Tính xác suất tiên nghiệm P(Ci) cho mỗi nhãn
        for label in priors:
            priors[label] /= len(labels)  # P(Ci)

        # Tìm nhãn có xác suất tiên nghiệm cao nhất
        max_posterior = max(priors, key=priors.get)
        #print(f"Gia truong co xac suat hau nghiem cao nhat trong cot {col_name}: {max_posterior}")
        return max_posterior



    def sum_of_columns(self,col_name):
        # Tính tổng các giá trị trong mỗi cột của DataFrame
        each_value_in_column = self.df[col_name].sum()
        print(f"Tổng các giá trị trong cột {col_name}: {each_value_in_column}")
        return each_value_in_column

    # Đếm số lượng mẫu của mỗi lớp
    from collections import Counter



    def __init__(self, prior=None, likelihood=None):
        """
        Initialize the Bayes model with a prior and likelihood.

        :param prior: The prior distribution.
        :param likelihood: The likelihood function.
        """
        self.prior = prior
        self.likelihood = likelihood

    def update(self, data):
        """
        Update the model with new data.

        :param data: The observed data.
        :return: Updated posterior distribution.
        """
        # Placeholder for update logic
        pass



