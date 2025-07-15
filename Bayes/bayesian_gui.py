import sys
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import traceback
from bayes_classifier import BayesianClassifier


class BayesianClassifierGUI(QtWidgets.QMainWindow):
    """Ứng dụng PyQt5 cho Bayesian Classification"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bayesian Classification Explorer")
        self.resize(1200, 600)

        # Khởi tạo mô hình
        self.bayes_model = BayesianClassifier()
        self.current_results = None

        # ---------- Central widget & layout ---------- #
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # 1 --- Chọn file CSV ------------------------------------------------
        file_layout = QtWidgets.QHBoxLayout()
        self.fileLine = QtWidgets.QLineEdit()
        self.fileLine.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Chọn file CSV…")
        browse_btn.clicked.connect(self.browse_file)

        file_layout.addWidget(QtWidgets.QLabel("Tập dữ liệu:"))
        file_layout.addWidget(self.fileLine)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        # 2 --- Chọn cột target và thông số -----------------------------
        param_layout = QtWidgets.QHBoxLayout()

        # Target column selection
        self.targetCombo = QtWidgets.QComboBox()
        self.targetCombo.setEnabled(False)

        # Test size selection
        self.testSizeSpin = QtWidgets.QDoubleSpinBox()
        self.testSizeSpin.setDecimals(2)
        self.testSizeSpin.setRange(0.1, 0.9)
        self.testSizeSpin.setSingleStep(0.05)
        self.testSizeSpin.setValue(0.2)

        # Random state
        self.randomStateSpin = QtWidgets.QSpinBox()
        self.randomStateSpin.setRange(0, 1000)
        self.randomStateSpin.setValue(42)

        param_layout.addWidget(QtWidgets.QLabel("Cột phân lớp:"))
        param_layout.addWidget(self.targetCombo)
        param_layout.addSpacing(20)
        param_layout.addWidget(QtWidgets.QLabel("Tỷ lệ test:"))
        param_layout.addWidget(self.testSizeSpin)
        param_layout.addSpacing(20)
        param_layout.addWidget(QtWidgets.QLabel("Random State:"))
        param_layout.addWidget(self.randomStateSpin)
        param_layout.addSpacing(30)

        run_btn = QtWidgets.QPushButton("Chạy phân lớp Bayes")
        run_btn.clicked.connect(self.run_classification)
        param_layout.addWidget(run_btn)

        layout.addLayout(param_layout)

        # 3 --- Thông tin dữ liệu ----------------------------------------
        info_layout = QtWidgets.QHBoxLayout()

        # Dataset info
        self.dataInfoBox = QtWidgets.QTextEdit()
        self.dataInfoBox.setReadOnly(True)
        self.dataInfoBox.setMaximumHeight(100)
        self.dataInfoBox.setPlainText("Thông tin dữ liệu sẽ hiển thị ở đây...")

        info_layout.addWidget(self.dataInfoBox)
        layout.addLayout(info_layout)

        # 4 --- Các vùng hiển thị kết quả ---
        result_layout_label = QtWidgets.QHBoxLayout()
        box_model_info_label = QtWidgets.QLabel("Thông tin mô hình:")
        box_evaluation_label = QtWidgets.QLabel("Đánh giá mô hình:")
        box_statistics_label = QtWidgets.QLabel("Thống kê dữ liệu:")

        result_layout_label.addWidget(box_model_info_label)
        result_layout_label.addWidget(box_evaluation_label)
        result_layout_label.addWidget(box_statistics_label)

        result_layout = QtWidgets.QHBoxLayout()
        self.box_model_info = QtWidgets.QTextEdit()
        self.box_model_info.setReadOnly(True)

        self.box_evaluation = QtWidgets.QTextEdit()
        self.box_evaluation.setReadOnly(True)

        self.box_statistics = QtWidgets.QTextEdit()
        self.box_statistics.setReadOnly(True)

        result_layout.addWidget(self.box_model_info)
        result_layout.addWidget(self.box_evaluation)
        result_layout.addWidget(self.box_statistics)

        layout.addLayout(result_layout_label)
        layout.addLayout(result_layout)

        # 5 --- Prediction section ----------------------------------------
        prediction_layout = QtWidgets.QHBoxLayout()

        predict_btn = QtWidgets.QPushButton("Dự đoán mẫu mới")
        predict_btn.clicked.connect(self.show_prediction_dialog)

        prediction_layout.addWidget(predict_btn)
        prediction_layout.addStretch()

        layout.addLayout(prediction_layout)

    # ------------------------------------------------------------------
    #                       SLOT: Browse file
    # ------------------------------------------------------------------
    def browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Chọn file CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            # Load dữ liệu bằng model
            data_info = self.bayes_model.load_data(path)

            # Hiển thị thông tin dữ liệu
            self.display_data_info(data_info)

            # Cập nhật combo box với các cột
            self.targetCombo.clear()
            self.targetCombo.addItems(data_info['columns'])
            self.targetCombo.setEnabled(True)

            # Cập nhật đường dẫn file
            self.fileLine.setText(path)

        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Lỗi đọc file", f"Không thể đọc file CSV:\n{str(exc)}")

    # ------------------------------------------------------------------
    #                       Display data info
    # ------------------------------------------------------------------
    def display_data_info(self, data_info):
        info_text = f"Kích thước dữ liệu: {data_info['shape'][0]} hàng × {data_info['shape'][1]} cột\n"
        info_text += f"Các cột: {', '.join(data_info['columns'])}\n"
        info_text += f"Dữ liệu thiếu: {data_info['missing_values']} ô\n"
        info_text += f"Kiểu dữ liệu: {len([k for k, v in data_info['data_types'].items() if v == 'object'])} categorical, "
        info_text += f"{len([k for k, v in data_info['data_types'].items() if v != 'object'])} numeric"

        self.dataInfoBox.setPlainText(info_text)

    # ------------------------------------------------------------------
    #                       SLOT: Run Classification
    # ------------------------------------------------------------------
    def run_classification(self):
        if self.bayes_model.df_raw is None:
            QtWidgets.QMessageBox.warning(self, "Chưa có dữ liệu", "Vui lòng chọn file CSV trước.")
            return

        target_column = self.targetCombo.currentText()
        if not target_column:
            QtWidgets.QMessageBox.warning(self, "Chưa chọn cột target", "Vui lòng chọn cột phân lớp.")
            return

        try:
            # Tham số
            test_size = self.testSizeSpin.value()
            random_state = self.randomStateSpin.value()

            # Chạy pipeline hoàn chỉnh
            self.current_results = self.bayes_model.run_complete_pipeline(
                self.fileLine.text(), target_column, test_size, random_state
            )

            # Hiển thị kết quả
            self.display_results()

        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Lỗi xử lý", f"Lỗi trong quá trình phân lớp:\n{str(exc)}")

    # ------------------------------------------------------------------
    #                       Display Results
    # ------------------------------------------------------------------
    def display_results(self):
        if not self.current_results:
            return

        # 1. Hiển thị thông tin mô hình
        self.display_model_info()

        # 2. Hiển thị đánh giá mô hình
        self.display_evaluation()

        # 3. Hiển thị thống kê
        self.display_statistics()

    def display_model_info(self):
        model_info = self.current_results['model_info']
        features_info = self.current_results['features_info']
        split_info = self.current_results['split_info']

        text = "=== THÔNG TIN MÔ HÌNH ===\n\n"
        text += f"Thuật toán: {model_info['model_type']}\n"
        text += f"Số lượng features: {model_info['n_features']}\n"
        text += f"Số lượng classes: {model_info['n_classes']}\n"
        text += f"Tập huấn luyện: {split_info['train_size']} mẫu\n"
        text += f"Tập kiểm tra: {split_info['test_size']} mẫu\n\n"

        text += f"Các lớp: {model_info['classes']}\n\n"

        text += "Xác suất tiên nghiệm:\n"
        for i, class_label in enumerate(model_info['classes']):
            text += f"  Class {class_label}: {model_info['class_prior'][i]:.4f}\n"

        text += f"\nFeatures: {', '.join(model_info['feature_names'])}\n"

        self.box_model_info.setText(text)

    def display_evaluation(self):
        eval_info = self.current_results['evaluation_info']

        text = "=== ĐÁNH GIÁ MÔ HÌNH ===\n\n"
        text += f"Độ chính xác: {eval_info['accuracy']:.4f} ({eval_info['accuracy'] * 100:.2f}%)\n\n"

        # Classification report
        text += "Chi tiết phân lớp:\n"
        for class_name, metrics in eval_info['classification_report'].items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                text += f"  Class {class_name}:\n"
                text += f"    Precision: {metrics['precision']:.3f}\n"
                text += f"    Recall: {metrics['recall']:.3f}\n"
                text += f"    F1-score: {metrics['f1-score']:.3f}\n"

        text += "\nMa trận nhầm lẫn:\n"
        cm = eval_info['confusion_matrix']
        for row in cm:
            text += f"  {row}\n"

        text += "\nMột số dự đoán mẫu:\n"
        for i, sample in enumerate(eval_info['sample_predictions'][:5]):
            text += f"  Mẫu {i + 1}: Thực tế={sample['actual']}, Dự đoán={sample['predicted']}, "
            text += f"Độ tin cậy={sample['probability']:.3f}\n"

        self.box_evaluation.setText(text)

    def display_statistics(self):
        stats = self.current_results['statistics']
        max_posterior = self.current_results['max_posterior_info']

        text = "=== THỐNG KÊ DỮ LIỆU ===\n\n"

        text += f"Lớp có xác suất tiên nghiệm cao nhất: {max_posterior['max_posterior_class']}\n\n"

        text += "Phân bố lớp:\n"
        for class_name, count in max_posterior['class_counts'].items():
            percentage = (count / sum(max_posterior['class_counts'].values())) * 100
            text += f"  {class_name}: {count} ({percentage:.1f}%)\n"

        text += "\nThống kê các cột:\n"
        for col, stat in stats.items():
            text += f"\n{col} ({stat['type']}):\n"
            if stat['type'] == 'numeric':
                text += f"  Trung bình: {stat['mean']:.3f}\n"
                text += f"  Độ lệch chuẩn: {stat['std']:.3f}\n"
                text += f"  Min: {stat['min']:.3f}, Max: {stat['max']:.3f}\n"
            else:
                text += f"  Giá trị duy nhất: {stat['unique_values']}\n"
                text += f"  Phổ biến nhất: {stat['most_frequent']}\n"

        text += "\n=== ĐỊNH LÝ BAYES ===\n"
        text += "P(Class|Features) = P(Features|Class) × P(Class) / P(Features)\n\n"
        text += "Trong đó:\n"
        text += "- P(Class|Features): Xác suất hậu nghiệm\n"
        text += "- P(Features|Class): Likelihood (khả năng)\n"
        text += "- P(Class): Xác suất tiên nghiệm\n"
        text += "- P(Features): Evidence (bằng chứng)\n"

        self.box_statistics.setText(text)

    # ------------------------------------------------------------------
    #                       Prediction Dialog
    # ------------------------------------------------------------------
    def show_prediction_dialog(self):
        if self.bayes_model.model is None:
            QtWidgets.QMessageBox.warning(self, "Chưa có mô hình", "Vui lòng huấn luyện mô hình trước.")
            return

        dialog = PredictionDialog(self.bayes_model, self)
        dialog.exec_()


class PredictionDialog(QtWidgets.QDialog):
    """Dialog để nhập dữ liệu cho dự đoán"""

    def __init__(self, bayes_model, parent=None):
        super().__init__(parent)
        self.bayes_model = bayes_model
        self.setWindowTitle("Dự đoán mẫu mới")
        self.setModal(True)
        self.resize(400, 300)

        layout = QtWidgets.QVBoxLayout(self)

        # Input fields
        self.input_fields = {}
        for feature in self.bayes_model.feature_columns:
            h_layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(f"{feature}:")
            input_field = QtWidgets.QLineEdit()

            h_layout.addWidget(label)
            h_layout.addWidget(input_field)
            layout.addLayout(h_layout)

            self.input_fields[feature] = input_field

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        predict_btn = QtWidgets.QPushButton("Dự đoán")
        predict_btn.clicked.connect(self.predict)
        cancel_btn = QtWidgets.QPushButton("Hủy")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(predict_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        # Result area
        self.result_text = QtWidgets.QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(100)
        layout.addWidget(self.result_text)

    def predict(self):
        try:
            # Collect input data
            sample_data = {}
            for feature, input_field in self.input_fields.items():
                value = input_field.text().strip()
                if not value:
                    QtWidgets.QMessageBox.warning(self, "Thiếu dữ liệu", f"Vui lòng nhập giá trị cho {feature}")
                    return

                # Try to convert to number if possible
                try:
                    sample_data[feature] = float(value)
                except ValueError:
                    sample_data[feature] = value

            # Make prediction
            result = self.bayes_model.predict_new_sample(sample_data)

            # Display result
            result_text = f"Dự đoán: {result['prediction']}\n"
            result_text += f"Độ tin cậy: {result['confidence']:.3f}\n"
            result_text += f"Xác suất cho các lớp: {result['probabilities']}"

            self.result_text.setPlainText(result_text)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi dự đoán", f"Lỗi khi dự đoán:\n{str(e)}")


# =============================== MAIN ===============================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show GUI
    gui = BayesianClassifierGUI()
    gui.show()

    sys.exit(app.exec_())