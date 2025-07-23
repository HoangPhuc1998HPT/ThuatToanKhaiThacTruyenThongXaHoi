import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import seaborn as sns
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSplitter
import warnings

warnings.filterwarnings('ignore')


class DecisionTreeGUI(QtWidgets.QMainWindow):
    """Ứng dụng PyQt5 cho Decision Tree Classification"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Decision Tree Explorer")
        self.resize(1400, 900)

        # Data holders
        self._df_raw = None
        self._model = None
        self._label_encoders = {}
        self._feature_names = None
        self._target_name = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện người dùng"""
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # === FILE SELECTION ===
        file_group = QtWidgets.QGroupBox("1. Chọn dữ liệu")
        file_layout = QtWidgets.QHBoxLayout(file_group)

        self.file_line = QtWidgets.QLineEdit()
        self.file_line.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Chọn file CSV...")
        browse_btn.clicked.connect(self.browse_file)

        file_layout.addWidget(QtWidgets.QLabel("File:"))
        file_layout.addWidget(self.file_line)
        file_layout.addWidget(browse_btn)

        main_layout.addWidget(file_group)

        # === DATA PREPROCESSING ===
        preprocess_group = QtWidgets.QGroupBox("2. Tiền xử lý dữ liệu")
        preprocess_layout = QtWidgets.QVBoxLayout(preprocess_group)

        # Target selection
        target_layout = QtWidgets.QHBoxLayout()
        self.target_combo = QtWidgets.QComboBox()
        analyze_btn = QtWidgets.QPushButton("Phân tích dữ liệu")
        analyze_btn.clicked.connect(self.analyze_data)

        target_layout.addWidget(QtWidgets.QLabel("Cột target:"))
        target_layout.addWidget(self.target_combo)
        target_layout.addWidget(analyze_btn)
        target_layout.addStretch()

        preprocess_layout.addLayout(target_layout)

        # Categorical columns selection
        cat_layout = QtWidgets.QHBoxLayout()
        self.cat_list = QtWidgets.QListWidget()
        self.cat_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.cat_list.setMaximumHeight(100)

        cat_layout.addWidget(QtWidgets.QLabel("Cột categorical:"))
        cat_layout.addWidget(self.cat_list)

        preprocess_layout.addLayout(cat_layout)

        main_layout.addWidget(preprocess_group)

        # === MODEL PARAMETERS ===
        param_group = QtWidgets.QGroupBox("3. Tham số mô hình")
        param_layout = QtWidgets.QHBoxLayout(param_group)

        # Criterion
        self.criterion_combo = QtWidgets.QComboBox()
        self.criterion_combo.addItems(["entropy", "gini"])

        # Max depth
        self.depth_spin = QtWidgets.QSpinBox()
        self.depth_spin.setRange(1, 20)
        self.depth_spin.setValue(4)

        # Test size
        self.test_size_spin = QtWidgets.QDoubleSpinBox()
        self.test_size_spin.setDecimals(2)
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.3)

        # Random state
        self.random_state_spin = QtWidgets.QSpinBox()
        self.random_state_spin.setRange(1, 1000)
        self.random_state_spin.setValue(42)

        param_layout.addWidget(QtWidgets.QLabel("Criterion:"))
        param_layout.addWidget(self.criterion_combo)
        param_layout.addWidget(QtWidgets.QLabel("Max Depth:"))
        param_layout.addWidget(self.depth_spin)
        param_layout.addWidget(QtWidgets.QLabel("Test Size:"))
        param_layout.addWidget(self.test_size_spin)
        param_layout.addWidget(QtWidgets.QLabel("Random State:"))
        param_layout.addWidget(self.random_state_spin)

        # Train button
        train_btn = QtWidgets.QPushButton("Huấn luyện mô hình")
        train_btn.clicked.connect(self.train_model)
        param_layout.addWidget(train_btn)

        main_layout.addWidget(param_group)

        # === RESULTS SECTION ===
        results_splitter = QSplitter(QtCore.Qt.Horizontal)

        # Left panel - Text results
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        # Data info
        self.data_info_text = QtWidgets.QTextEdit()
        self.data_info_text.setMaximumHeight(150)
        self.data_info_text.setReadOnly(True)
        left_layout.addWidget(QtWidgets.QLabel("Thông tin dữ liệu:"))
        left_layout.addWidget(self.data_info_text)

        # Model results
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        left_layout.addWidget(QtWidgets.QLabel("Kết quả mô hình:"))
        left_layout.addWidget(self.results_text)

        results_splitter.addWidget(left_widget)

        # Right panel - Visualizations
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        # Visualization tabs
        self.viz_tabs = QtWidgets.QTabWidget()

        # Tab 1: Tree visualization
        self.tree_canvas = PlotCanvas()
        self.viz_tabs.addTab(self.tree_canvas, "Cây quyết định")

        # Tab 2: Feature importance
        self.importance_canvas = PlotCanvas()
        self.viz_tabs.addTab(self.importance_canvas, "Độ quan trọng")

        # Tab 3: Target distribution
        self.dist_canvas = PlotCanvas()
        self.viz_tabs.addTab(self.dist_canvas, "Phân phối target")

        right_layout.addWidget(self.viz_tabs)
        results_splitter.addWidget(right_widget)

        main_layout.addWidget(results_splitter)

        # === PREDICTION SECTION - SỬA LẠI PHẦN NÀY ===
        pred_group = QtWidgets.QGroupBox("4. Dự đoán")
        pred_group.setMaximumHeight(200)  # Giới hạn chiều cao
        pred_main_layout = QtWidgets.QVBoxLayout(pred_group)

        # Input section trong một horizontal layout
        input_section = QtWidgets.QWidget()
        input_layout = QtWidgets.QHBoxLayout(input_section)

        # Left side - Input fields
        input_left = QtWidgets.QWidget()
        input_left_layout = QtWidgets.QVBoxLayout(input_left)
        input_left_layout.addWidget(QtWidgets.QLabel("Nhập dữ liệu:"))

        self.pred_inputs = {}
        self.pred_scroll = QtWidgets.QScrollArea()
        self.pred_scroll.setMaximumHeight(120)  # Giới hạn chiều cao scroll area
        self.pred_scroll.setWidgetResizable(True)
        self.pred_widget = QtWidgets.QWidget()
        self.pred_layout = QtWidgets.QFormLayout(self.pred_widget)
        self.pred_scroll.setWidget(self.pred_widget)

        input_left_layout.addWidget(self.pred_scroll)

        pred_btn = QtWidgets.QPushButton("Dự đoán")
        pred_btn.clicked.connect(self.make_prediction)
        input_left_layout.addWidget(pred_btn)

        input_layout.addWidget(input_left)

        # Right side - Results
        result_right = QtWidgets.QWidget()
        result_layout = QtWidgets.QVBoxLayout(result_right)
        result_layout.addWidget(QtWidgets.QLabel("Kết quả dự đoán:"))

        # SỬA: Sử dụng QTextEdit thay vì QLabel để hiển thị tốt hơn
        self.pred_result = QtWidgets.QTextEdit()
        self.pred_result.setReadOnly(True)
        self.pred_result.setMaximumHeight(120)
        self.pred_result.setText("Kết quả dự đoán sẽ hiển thị ở đây")

        result_layout.addWidget(self.pred_result)
        input_layout.addWidget(result_right)

        pred_main_layout.addWidget(input_section)
        main_layout.addWidget(pred_group)

    def browse_file(self):
        """Chọn file dữ liệu"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn file CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            # Try different separators
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(path, sep=sep)
                    if len(df.columns) > 1:
                        self._df_raw = df
                        break
                except:
                    continue

            if self._df_raw is None:
                raise ValueError("Không thể đọc file với các định dạng thông dụng")

            self.file_line.setText(path)

            # Populate combo boxes
            self.target_combo.clear()
            self.target_combo.addItems(self._df_raw.columns.tolist())

            self.cat_list.clear()
            for col in self._df_raw.columns:
                item = QtWidgets.QListWidgetItem(col)
                self.cat_list.addItem(item)
                if self._df_raw[col].dtype == 'object':
                    item.setSelected(True)

            QtWidgets.QMessageBox.information(
                self, "Thành công",
                f"Đã tải dữ liệu: {self._df_raw.shape[0]} dòng, {self._df_raw.shape[1]} cột"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Lỗi đọc file: {str(e)}")

    def analyze_data(self):
        """Phân tích dữ liệu"""
        if self._df_raw is None:
            QtWidgets.QMessageBox.warning(self, "Cảnh báo", "Chưa chọn dữ liệu")
            return

        try:
            # Basic info
            info_text = f"=== THÔNG TIN DỮ LIỆU ===\n"
            info_text += f"Shape: {self._df_raw.shape}\n"
            info_text += f"Columns: {list(self._df_raw.columns)}\n\n"

            # Missing values
            missing = self._df_raw.isnull().sum()
            info_text += f"=== GIÁ TRỊ THIẾU ===\n"
            if missing.sum() == 0:
                info_text += "Không có giá trị thiếu\n\n"
            else:
                for col, count in missing[missing > 0].items():
                    info_text += f"{col}: {count}\n"
                info_text += "\n"

            # Data types
            info_text += f"=== KIỂU DỮ LIỆU ===\n"
            for col, dtype in self._df_raw.dtypes.items():
                info_text += f"{col}: {dtype}\n"

            self.data_info_text.setText(info_text)

            # Plot target distribution if target is selected
            target_col = self.target_combo.currentText()
            if target_col:
                self.plot_target_distribution(target_col)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Lỗi phân tích: {str(e)}")

    def plot_target_distribution(self, target_col):
        """Vẽ phân phối target"""
        try:
            self.dist_canvas.figure.clear()
            ax = self.dist_canvas.figure.add_subplot(111)

            value_counts = self._df_raw[target_col].value_counts()
            ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_xlabel(target_col)
            ax.set_ylabel('Số lượng')
            ax.set_title(f'Phân phối {target_col}')
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45)

            self.dist_canvas.figure.tight_layout()
            self.dist_canvas.draw()

        except Exception as e:
            print(f"Lỗi vẽ biểu đồ: {e}")

    def train_model(self):
        """Huấn luyện mô hình"""
        if self._df_raw is None:
            QtWidgets.QMessageBox.warning(self, "Cảnh báo", "Chưa chọn dữ liệu")
            return

        try:
            # Get parameters
            target_col = self.target_combo.currentText()
            if not target_col:
                QtWidgets.QMessageBox.warning(self, "Cảnh báo", "Chưa chọn cột target")
                return

            self._target_name = target_col

            # Get categorical columns
            cat_cols = [item.text() for item in self.cat_list.selectedItems()]
            if target_col in cat_cols:
                cat_cols.remove(target_col)

            # Prepare data
            df_work = self._df_raw.copy()

            # Encode categorical features
            self._label_encoders = {}
            for col in cat_cols:
                if col in df_work.columns:
                    le = LabelEncoder()
                    df_work[col] = le.fit_transform(df_work[col].astype(str))
                    self._label_encoders[col] = le

            # Prepare features and target
            X = df_work.drop(columns=[target_col])
            y = df_work[target_col]
            self._feature_names = X.columns.tolist()

            # Split data
            test_size = self.test_size_spin.value()
            random_state = self.random_state_spin.value()

            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Train model
            criterion = self.criterion_combo.currentText()
            max_depth = self.depth_spin.value()

            self._model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                random_state=random_state
            )

            self._model.fit(self._X_train, self._y_train)

            # Evaluate
            train_pred = self._model.predict(self._X_train)
            test_pred = self._model.predict(self._X_test)

            train_acc = metrics.accuracy_score(self._y_train, train_pred)
            test_acc = metrics.accuracy_score(self._y_test, test_pred)

            # Display results
            results_text = f"=== KẾT QUẢ HUẤN LUYỆN ===\n"
            results_text += f"Tham số:\n"
            results_text += f"  - Criterion: {criterion}\n"
            results_text += f"  - Max depth: {max_depth}\n"
            results_text += f"  - Test size: {test_size}\n\n"
            results_text += f"Kết quả:\n"
            results_text += f"  - Training accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)\n"
            results_text += f"  - Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)\n\n"

            # Classification report
            results_text += f"=== CLASSIFICATION REPORT ===\n"
            results_text += metrics.classification_report(self._y_test, test_pred)

            self.results_text.setText(results_text)

            # Plot tree
            self.plot_tree()

            # Plot feature importance
            self.plot_feature_importance()

            # Setup prediction inputs
            self.setup_prediction_inputs()

            QtWidgets.QMessageBox.information(
                self, "Thành công",
                f"Mô hình đã được huấn luyện!\nTest accuracy: {test_acc:.4f}"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Lỗi huấn luyện: {str(e)}")

    def plot_tree(self):
        """Vẽ cây quyết định"""
        if self._model is None:
            return

        try:
            self.tree_canvas.figure.clear()
            ax = self.tree_canvas.figure.add_subplot(111)

            plot_tree(self._model,
                      feature_names=self._feature_names,
                      class_names=[str(c) for c in self._model.classes_],
                      filled=True,
                      rounded=True,
                      ax=ax)

            ax.set_title("Decision Tree")
            self.tree_canvas.figure.tight_layout()
            self.tree_canvas.draw()

        except Exception as e:
            print(f"Lỗi vẽ cây: {e}")

    def plot_feature_importance(self):
        """Vẽ độ quan trọng của features"""
        if self._model is None:
            return

        try:
            self.importance_canvas.figure.clear()
            ax = self.importance_canvas.figure.add_subplot(111)

            importance = self._model.feature_importances_
            indices = np.argsort(importance)[::-1]

            ax.bar(range(len(importance)), importance[indices])
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importance')
            ax.set_xticks(range(len(importance)))
            ax.set_xticklabels([self._feature_names[i] for i in indices], rotation=45)

            self.importance_canvas.figure.tight_layout()
            self.importance_canvas.draw()

        except Exception as e:
            print(f"Lỗi vẽ feature importance: {e}")

    def setup_prediction_inputs(self):
        """Thiết lập các trường input cho dự đoán"""
        # Clear existing inputs
        for i in reversed(range(self.pred_layout.count())):
            child = self.pred_layout.itemAt(i).widget()
            if child:
                child.deleteLater()

        self.pred_inputs = {}

        for feature in self._feature_names:
            if feature in self._label_encoders:
                # Categorical feature - use combo box
                combo = QtWidgets.QComboBox()
                combo.addItems(self._label_encoders[feature].classes_)
                self.pred_inputs[feature] = combo
                self.pred_layout.addRow(f"{feature}:", combo)
            else:
                # Numerical feature - use line edit
                line_edit = QtWidgets.QLineEdit()
                line_edit.setPlaceholderText("Nhập giá trị số")
                self.pred_inputs[feature] = line_edit
                self.pred_layout.addRow(f"{feature}:", line_edit)

    def make_prediction(self):
        """Thực hiện dự đoán"""
        if self._model is None:
            QtWidgets.QMessageBox.warning(self, "Cảnh báo", "Chưa huấn luyện mô hình")
            return

        try:
            # Collect input values
            input_data = {}
            for feature, widget in self.pred_inputs.items():
                if isinstance(widget, QtWidgets.QComboBox):
                    # Categorical feature
                    value = widget.currentText()
                    encoded_value = self._label_encoders[feature].transform([value])[0]
                    input_data[feature] = encoded_value
                else:
                    # Numerical feature
                    try:
                        text_value = widget.text().strip()
                        if not text_value:
                            QtWidgets.QMessageBox.warning(
                                self, "Lỗi", f"Vui lòng nhập giá trị cho {feature}"
                            )
                            return
                        value = float(text_value)
                        input_data[feature] = value
                    except ValueError:
                        QtWidgets.QMessageBox.warning(
                            self, "Lỗi", f"Giá trị không hợp lệ cho {feature}"
                        )
                        return

            # Make prediction
            input_df = pd.DataFrame([input_data])
            prediction = self._model.predict(input_df)[0]
            probabilities = self._model.predict_proba(input_df)[0]

            # SỬA: Format kết quả hiển thị rõ ràng hơn
            result_text = f"=== KẾT QUẢ DỰ ĐOÁN ===\n\n"
            result_text += f"Dự đoán: {prediction}\n\n"
            result_text += "Xác suất cho từng lớp:\n"

            # Sắp xếp theo xác suất giảm dần
            prob_pairs = list(zip(self._model.classes_, probabilities))
            prob_pairs.sort(key=lambda x: x[1], reverse=True)

            for class_name, prob in prob_pairs:
                result_text += f"  • {class_name}: {prob:.4f} ({prob * 100:.2f}%)\n"

            # Thêm thông tin input đã nhập
            result_text += f"\n=== DỮ LIỆU NHẬP ===\n"
            for feature, widget in self.pred_inputs.items():
                if isinstance(widget, QtWidgets.QComboBox):
                    value = widget.currentText()
                else:
                    value = widget.text()
                result_text += f"  • {feature}: {value}\n"

            self.pred_result.setText(result_text)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Lỗi dự đoán: {str(e)}")
            # Clear result on error
            self.pred_result.setText("Có lỗi xảy ra trong quá trình dự đoán")


class PlotCanvas(FigureCanvas):
    """Canvas để hiển thị matplotlib plots"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)


# =============================== MAIN ===============================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = DecisionTreeGUI()
    gui.show()
    sys.exit(app.exec_())