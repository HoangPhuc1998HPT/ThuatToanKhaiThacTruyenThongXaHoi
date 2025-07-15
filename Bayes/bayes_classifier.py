import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class BayesianClassifier:
    """
    Class để xây dựng và quản lý mô hình Bayesian Classification
    """

    def __init__(self):
        """
        Khởi tạo mô hình Bayesian Classifier
        """
        self.df_raw = None
        self.df_clean = None
        self.label_encoders = {}
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_column = None
        self.feature_columns = None
        self.predictions = None
        self.prediction_probabilities = None

    def load_data(self, file_path):
        """
        Tải dữ liệu từ file CSV

        Args:
            file_path (str): Đường dẫn đến file CSV

        Returns:
            dict: Thông tin về dữ liệu đã tải
        """
        try:
            self.df_raw = pd.read_csv(file_path)

            # Thông tin dữ liệu
            info = {
                'shape': self.df_raw.shape,
                'columns': self.df_raw.columns.tolist(),
                'missing_values': self.df_raw.isnull().sum().sum(),
                'data_types': self.df_raw.dtypes.to_dict(),
                'sample_data': self.df_raw.head().to_dict()
            }

            return info

        except Exception as e:
            raise Exception(f"Không thể tải dữ liệu: {str(e)}")

    def clean_data(self):
        """
        Làm sạch dữ liệu: loại bỏ NaN, duplicate, reset index

        Returns:
            dict: Thông tin về quá trình làm sạch
        """
        if self.df_raw is None:
            raise Exception("Chưa tải dữ liệu")

        original_shape = self.df_raw.shape

        # Làm sạch dữ liệu
        self.df_clean = self.df_raw.copy()
        self.df_clean = self.df_clean.dropna()
        self.df_clean = self.df_clean.drop_duplicates()
        self.df_clean = self.df_clean.reset_index(drop=True)
        clean_info = {
            'original_shape': original_shape,
            'clean_shape': self.df_clean.shape,
            'removed_rows': original_shape[0] - self.df_clean.shape[0],
            'removed_percentage': ((original_shape[0] - self.df_clean.shape[0]) / original_shape[0]) * 100
        }

        return clean_info

    def encode_categorical_data(self):
        """
        Encode các biến categorical thành số

        Returns:
            dict: Thông tin về quá trình encoding
        """
        if self.df_clean is None:
            raise Exception("Chưa làm sạch dữ liệu")

        df_encoded = self.df_clean.copy()
        encoded_columns = []

        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                encoded_columns.append(col)

        self.df_clean = df_encoded

        encoding_info = {
            'encoded_columns': encoded_columns,
            'total_encoded': len(encoded_columns),
            'encoders': {col: list(self.label_encoders[col].classes_) for col in encoded_columns}
        }

        return encoding_info

    def prepare_features_target(self, target_column):
        """
        Chuẩn bị features và target cho training

        Args:
            target_column (str): Tên cột target

        Returns:
            dict: Thông tin về features và target
        """
        if self.df_clean is None:
            raise Exception("Chưa xử lý dữ liệu")

        if target_column not in self.df_clean.columns:
            raise Exception(f"Cột '{target_column}' không tồn tại")

        self.target_column = target_column
        self.feature_columns = [col for col in self.df_clean.columns if col != target_column]

        X = self.df_clean[self.feature_columns]
        y = self.df_clean[target_column]

        features_info = {
            'target_column': target_column,
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns),
            'n_samples': len(X),
            'target_classes': np.unique(y).tolist(),
            'n_classes': len(np.unique(y)),
            'class_distribution': Counter(y)
        }

        return features_info

    def split_data(self, test_size=0.2, random_state=42):
        """
        Chia dữ liệu thành tập train và test

        Args:
            test_size (float): Tỷ lệ dữ liệu test
            random_state (int): Seed cho random

        Returns:
            dict: Thông tin về việc chia dữ liệu
        """


        if self.target_column is None:
            raise Exception("Chưa chuẩn bị features và target")

        X = self.df_clean[self.feature_columns]
        y = self.df_clean[self.target_column]

        self.check_class_balance(self.df_clean, y)
        # Chia dữ liệu
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        split_info = {
            'test_size': test_size,
            'random_state': random_state,
            'train_size': len(self.X_train),
            'len_test_size': len(self.X_test),
            'train_class_distribution': Counter(self.y_train),
            'test_class_distribution': Counter(self.y_test)
        }

        return split_info

    def train_model(self):
        """
        Huấn luyện mô hình Gaussian Naive Bayes

        Returns:
            dict: Thông tin về mô hình đã huấn luyện
        """
        if self.X_train is None:
            raise Exception("Chưa chia dữ liệu train/test")

        # Khởi tạo và huấn luyện mô hình
        self.model = GaussianNB()
        self.model.fit(self.X_train, self.y_train)

        # Thông tin mô hình
        model_info = {
            'model_type': 'Gaussian Naive Bayes',
            'n_features': self.X_train.shape[1],
            'n_classes': len(self.model.classes_),
            'classes': self.model.classes_.tolist(),
            'class_prior': self.model.class_prior_.tolist(),
            'feature_names': self.feature_columns,
            'theta': self.model.theta_.tolist(),  # Mean của mỗi feature cho mỗi class
            'sigma': self.model.var_.tolist()  # Variance của mỗi feature cho mỗi class
        }

        return model_info

    def predict(self):
        """
        Dự đoán trên tập test

        Returns:
            dict: Kết quả dự đoán
        """
        if self.model is None:
            raise Exception("Chưa huấn luyện mô hình")

        # Dự đoán
        self.predictions = self.model.predict(self.X_test)
        self.prediction_probabilities = self.model.predict_proba(self.X_test)

        prediction_info = {
            'predictions': self.predictions.tolist(),
            'probabilities': self.prediction_probabilities.tolist(),
            'n_predictions': len(self.predictions)
        }

        return prediction_info

    def evaluate_model(self):
        """
        Đánh giá mô hình

        Returns:
            dict: Các metrics đánh giá
        """
        if self.predictions is None:
            raise Exception("Chưa thực hiện dự đoán")

        # Tính toán metrics
        accuracy = accuracy_score(self.y_test, self.predictions)
        classification_rep = classification_report(self.y_test, self.predictions, output_dict=True)
        conf_matrix = confusion_matrix(self.y_test, self.predictions)

        # Một số dự đoán mẫu
        sample_predictions = []
        for i in range(min(10, len(self.predictions))):
            sample_predictions.append({
                'actual': int(self.y_test.iloc[i]),
                'predicted': int(self.predictions[i]),
                'probability': float(self.prediction_probabilities[i].max()),
                'all_probabilities': self.prediction_probabilities[i].tolist()
            })

        evaluation_info = {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'sample_predictions': sample_predictions
        }

        return evaluation_info

    def calculate_posterior_probability(self, p_likelihood, p_prior, p_evidence):
        """
        Tính xác suất hậu nghiệm theo định lý Bayes

        Args:
            p_likelihood (float): P(X|Class) - Likelihood
            p_prior (float): P(Class) - Prior probability
            p_evidence (float): P(X) - Evidence

        Returns:
            float: P(Class|X) - Posterior probability
        """
        if p_evidence == 0:
            raise ValueError("P(X) không thể bằng 0")

        return (p_likelihood * p_prior) / p_evidence

    def get_max_posterior_class(self):
        """
        Tìm lớp có xác suất hậu nghiệm cao nhất (dựa trên prior)

        Returns:
            dict: Thông tin về lớp có xác suất cao nhất
        """
        if self.target_column is None:
            raise Exception("Chưa xác định target column")

        class_counts = Counter(self.df_clean[self.target_column])
        total_samples = len(self.df_clean)

        # Tính prior probability cho mỗi class
        class_priors = {cls: count / total_samples for cls, count in class_counts.items()}

        # Tìm class có prior cao nhất
        max_class = max(class_priors, key=class_priors.get)

        return {
            'max_posterior_class': max_class,
            'class_priors': class_priors,
            'class_counts': dict(class_counts)
        }

    def get_column_statistics(self):
        """
        Tính toán thống kê cho các cột

        Returns:
            dict: Thống kê chi tiết cho mỗi cột
        """
        if self.df_clean is None:
            raise Exception("Chưa có dữ liệu để tính thống kê")

        statistics = {}

        # Thống kê cho các cột số
        numeric_columns = self.df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            statistics[col] = {
                'type': 'numeric',
                'mean': float(self.df_clean[col].mean()),
                'std': float(self.df_clean[col].std()),
                'min': float(self.df_clean[col].min()),
                'max': float(self.df_clean[col].max()),
                'median': float(self.df_clean[col].median()),
                'missing_count': int(self.df_clean[col].isnull().sum())
            }

        # Thống kê cho các cột categorical
        categorical_columns = self.df_clean.select_dtypes(exclude=[np.number]).columns
        for col in categorical_columns:
            value_counts = self.df_clean[col].value_counts()
            statistics[col] = {
                'type': 'categorical',
                'unique_values': int(self.df_clean[col].nunique()),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'value_counts': value_counts.to_dict(),
                'missing_count': int(self.df_clean[col].isnull().sum())
            }

        return statistics

    def run_complete_pipeline(self, file_path, target_column, test_size=0.2, random_state=42):
        """
        Chạy toàn bộ pipeline từ đầu đến cuối

        Args:
            file_path (str): Đường dẫn file CSV
            target_column (str): Tên cột target
            test_size (float): Tỷ lệ dữ liệu test
            random_state (int): Seed cho random

        Returns:
            dict: Kết quả của toàn bộ pipeline
        """
        results = {}

        # 1. Load data
        results['data_info'] = self.load_data(file_path)

        # 2. Clean data
        results['clean_info'] = self.clean_data()

        # 3. Encode categorical data
        results['encoding_info'] = self.encode_categorical_data()

        # 4. Prepare features and target
        results['features_info'] = self.prepare_features_target(target_column)

        # 5. Split data
        results['split_info'] = self.split_data(test_size, random_state)

        # 6. Train model
        results['model_info'] = self.train_model()

        # 7. Predict
        results['prediction_info'] = self.predict()

        # 8. Evaluate
        results['evaluation_info'] = self.evaluate_model()

        # 9. Calculate statistics
        results['statistics'] = self.get_column_statistics()

        # 10. Get max posterior class
        results['max_posterior_info'] = self.get_max_posterior_class()

        return results

    def predict_new_sample(self, sample_data):
        """
        Dự đoán cho một mẫu mới

        Args:
            sample_data (dict): Dữ liệu mẫu mới

        Returns:
            dict: Kết quả dự đoán
        """
        if self.model is None:
            raise Exception("Chưa huấn luyện mô hình")

        # Chuyển đổi dữ liệu
        sample_df = pd.DataFrame([sample_data])

        # Encode categorical variables
        for col in sample_df.columns:
            if col in self.label_encoders:
                sample_df[col] = self.label_encoders[col].transform(sample_df[col].astype(str))

        # Đảm bảo thứ tự cột giống với training data
        sample_df = sample_df[self.feature_columns]

        # Dự đoán
        prediction = self.model.predict(sample_df)[0]
        probabilities = self.model.predict_proba(sample_df)[0]

        return {
            'prediction': int(prediction),
            'probabilities': probabilities.tolist(),
            'confidence': float(probabilities.max())
        }

    from collections import Counter

    def check_class_balance(self, df, y):
        """
        Kiểm tra xem có lớp nào có ít hơn 2 mẫu không
        """
        class_counts = Counter(y)
        low_classes = {cls: cnt for cls, cnt in class_counts.items() if cnt < 2}
        if low_classes:
            raise ValueError(
                f"Lỗi: Một số lớp có ít hơn 2 mẫu, không thể dùng stratify. Các lớp bị lỗi: {low_classes}"
            )

