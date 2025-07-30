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
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QSplitter
import warnings
import math

warnings.filterwarnings('ignore')


class ID3AnalyzerGUI(QtWidgets.QMainWindow):
    """Ứng dụng PyQt5 cho Decision Tree Classification với hiển thị quá trình ID3"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ID3 Decision Tree Explorer")
        self.resize(1600, 1000)

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

        # ID3 analysis data
        self._id3_analysis = []
        self._entropy_analysis = []

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
        param_layout = QtWidgets.QVBoxLayout(param_group)  # Đổi thành VBoxLayout

        # Dòng đầu tiên - các tham số
        param_row1 = QtWidgets.QHBoxLayout()

        # Criterion - FIXED to entropy for ID3
        self.criterion_combo = QtWidgets.QComboBox()
        self.criterion_combo.addItems(["entropy", "gini"])
        self.criterion_combo.setCurrentText("entropy")

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

        param_row1.addWidget(QtWidgets.QLabel("Criterion:"))
        param_row1.addWidget(self.criterion_combo)
        param_row1.addWidget(QtWidgets.QLabel("Max Depth:"))
        param_row1.addWidget(self.depth_spin)
        param_row1.addWidget(QtWidgets.QLabel("Test Size:"))
        param_row1.addWidget(self.test_size_spin)
        param_row1.addWidget(QtWidgets.QLabel("Random State:"))
        param_row1.addWidget(self.random_state_spin)

        # Dòng thứ hai - các nút điều khiển
        param_row2 = QtWidgets.QHBoxLayout()

        # Train button
        train_btn = QtWidgets.QPushButton("🚀 Huấn luyện mô hình & Phân tích ID3")
        train_btn.clicked.connect(self.train_model)
        param_row2.addWidget(train_btn)

        # THÊM CHECKBOX CHO CHẾ ĐỘ HIỂN THỊ CÂY
        self.detailed_tree_check = QtWidgets.QCheckBox("📋 Hiển thị cây với chú thích chi tiết")
        self.detailed_tree_check.setChecked(True)
        self.detailed_tree_check.stateChanged.connect(self.update_tree_display)
        param_row2.addWidget(self.detailed_tree_check)

        param_layout.addLayout(param_row1)
        param_layout.addLayout(param_row2)

        main_layout.addWidget(param_group)

        # === RESULTS SECTION ===
        results_splitter = QSplitter(QtCore.Qt.Horizontal)

        # Left panel - Text results với tabs
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        # Tạo tab widget cho kết quả text
        self.text_tabs = QtWidgets.QTabWidget()

        # Tab 1: Data info
        self.data_info_text = QtWidgets.QTextEdit()
        self.data_info_text.setReadOnly(True)
        self.text_tabs.addTab(self.data_info_text, "Thông tin dữ liệu")

        # Tab 2: ID3 Analysis - THÊM MỚI
        self.id3_analysis_text = QtWidgets.QTextEdit()
        self.id3_analysis_text.setReadOnly(True)
        self.text_tabs.addTab(self.id3_analysis_text, "Phân tích ID3")

        # Tab 3: Model results
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        self.text_tabs.addTab(self.results_text, "Kết quả mô hình")

        left_layout.addWidget(self.text_tabs)
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

        # Tab 4: Information Gain Chart - THÊM MỚI
        self.gain_canvas = PlotCanvas()
        self.viz_tabs.addTab(self.gain_canvas, "Information Gain")

        right_layout.addWidget(self.viz_tabs)
        results_splitter.addWidget(right_widget)

        main_layout.addWidget(results_splitter)

        # === PREDICTION SECTION ===
        pred_group = QtWidgets.QGroupBox("4. Dự đoán")
        pred_group.setMaximumHeight(200)
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
        self.pred_scroll.setMaximumHeight(120)
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

        self.pred_result = QtWidgets.QTextEdit()
        self.pred_result.setReadOnly(True)
        self.pred_result.setMaximumHeight(120)
        self.pred_result.setText("Kết quả dự đoán sẽ hiển thị ở đây")

        result_layout.addWidget(self.pred_result)
        input_layout.addWidget(result_right)

        pred_main_layout.addWidget(input_section)
        main_layout.addWidget(pred_group)

    def calculate_entropy(self, y):
        """Tính entropy của một tập dữ liệu"""
        if len(y) == 0:
            return 0

        value_counts = pd.Series(y).value_counts()
        probabilities = value_counts / len(y)
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        return entropy

    def calculate_information_gain(self, X, y, feature):
        """Tính Information Gain cho một feature"""
        # Entropy ban đầu
        initial_entropy = self.calculate_entropy(y)

        # Tính entropy có trọng số sau khi chia theo feature
        feature_values = X[feature].unique()
        weighted_entropy = 0
        split_info = []

        for value in feature_values:
            subset_mask = X[feature] == value
            subset_y = y[subset_mask]
            subset_size = len(subset_y)
            subset_entropy = self.calculate_entropy(subset_y)

            weight = subset_size / len(y)
            weighted_entropy += weight * subset_entropy

            split_info.append({
                'value': value,
                'size': subset_size,
                'entropy': subset_entropy,
                'weight': weight
            })

        information_gain = initial_entropy - weighted_entropy

        return {
            'initial_entropy': initial_entropy,
            'weighted_entropy': weighted_entropy,
            'information_gain': information_gain,
            'split_info': split_info
        }

    def analyze_id3_process(self, X, y):
        """Phân tích quá trình lựa chọn thuộc tính theo ID3"""
        self._id3_analysis = []
        self._entropy_analysis = []

        # Tính entropy ban đầu
        initial_entropy = self.calculate_entropy(y)
        self._entropy_analysis.append({
            'stage': 'Initial Dataset',
            'size': len(y),
            'entropy': initial_entropy,
            'class_distribution': pd.Series(y).value_counts().to_dict()
        })

        # Tính Information Gain cho tất cả features
        feature_gains = {}
        feature_details = {}

        for feature in X.columns:
            gain_info = self.calculate_information_gain(X, y, feature)
            feature_gains[feature] = gain_info['information_gain']
            feature_details[feature] = gain_info

        # Sắp xếp theo Information Gain giảm dần
        sorted_features = sorted(feature_gains.items(), key=lambda x: x[1], reverse=True)

        self._id3_analysis.append({
            'stage': 'Root Node Selection',
            'feature_gains': feature_gains,
            'feature_details': feature_details,
            'best_feature': sorted_features[0][0] if sorted_features else None,
            'best_gain': sorted_features[0][1] if sorted_features else 0
        })

        return sorted_features[0][0] if sorted_features else None

    def update_tree_display(self):
        """Cập nhật hiển thị cây theo chế độ được chọn"""
        if self._model is not None:
            if self.detailed_tree_check.isChecked():
                self.plot_tree_detailed()
            else:
                self.plot_tree()


    def display_id3_analysis(self):
        """Hiển thị kết quả phân tích ID3"""
        if not self._id3_analysis:
            return

        analysis_text = "🌳 PHÂN TÍCH QUÁ TRÌNH ID3 DECISION TREE\n"
        analysis_text += "=" * 70 + "\n\n"

        # Hiển thị entropy ban đầu
        if self._entropy_analysis:
            entropy_info = self._entropy_analysis[0]
            analysis_text += f"📊 BƯỚC 1: TÍNH ENTROPY BAN ĐẦU\n"
            analysis_text += f"{'=' * 50}\n"
            analysis_text += f"Tổng số mẫu: {entropy_info['size']}\n"
            analysis_text += f"Phân phối lớp:\n"

            for class_label, count in entropy_info['class_distribution'].items():
                probability = count / entropy_info['size']
                analysis_text += f"  • {class_label}: {count} mẫu ({probability:.3f})\n"

            analysis_text += f"\n📈 Entropy ban đầu: H(S) = {entropy_info['entropy']:.4f}\n"
            analysis_text += f"Công thức: H(S) = -Σ p_i * log2(p_i)\n\n"

        # Hiển thị phân tích từng feature
        for analysis in self._id3_analysis:
            if analysis['stage'] == 'Root Node Selection':
                analysis_text += f"🔍 BƯỚC 2: TÍNH INFORMATION GAIN CHO TỪNG THUỘC TÍNH\n"
                analysis_text += f"{'=' * 60}\n"

                # Sắp xếp features theo gain giảm dần
                sorted_gains = sorted(analysis['feature_gains'].items(),
                                      key=lambda x: x[1], reverse=True)

                for i, (feature, gain) in enumerate(sorted_gains, 1):
                    details = analysis['feature_details'][feature]
                    analysis_text += f"\n{i}. THUỘC TÍNH: {feature.upper()}\n"
                    analysis_text += f"   {'-' * 40}\n"

                    # Chi tiết chia theo từng giá trị
                    analysis_text += f"   Entropy sau khi chia:\n"
                    for split in details['split_info']:
                        analysis_text += f"     • {feature} = {split['value']}: "
                        analysis_text += f"{split['size']} mẫu, "
                        analysis_text += f"Entropy = {split['entropy']:.4f}, "
                        analysis_text += f"Trọng số = {split['weight']:.3f}\n"

                    analysis_text += f"\n   📊 Tính toán:\n"
                    analysis_text += f"     • Entropy có trọng số: {details['weighted_entropy']:.4f}\n"
                    analysis_text += f"     • Information Gain = {details['initial_entropy']:.4f} - {details['weighted_entropy']:.4f}\n"
                    analysis_text += f"     • Information Gain = {gain:.4f}\n"

                # Kết luận lựa chọn
                analysis_text += f"\n🎯 KẾT QUẢ LỰA CHỌN:\n"
                analysis_text += f"{'=' * 50}\n"
                analysis_text += f"Thuộc tính được chọn: {analysis['best_feature']}\n"
                analysis_text += f"Information Gain cao nhất: {analysis['best_gain']:.4f}\n\n"

                # Ranking table
                analysis_text += f"📋 BẢNG XẾP HẠNG THUỘC TÍNH:\n"
                analysis_text += f"{'=' * 50}\n"
                analysis_text += f"{'Hạng':<5} {'Thuộc tính':<15} {'Info Gain':<12} {'Ghi chú'}\n"
                analysis_text += f"{'-' * 50}\n"

                for i, (feature, gain) in enumerate(sorted_gains, 1):
                    note = "⭐ ĐƯỢC CHỌN" if i == 1 else ""
                    analysis_text += f"{i:<5} {feature:<15} {gain:<12.4f} {note}\n"

                analysis_text += f"\n"

        # Thêm giải thích thuật toán
        analysis_text += f"📚 GIẢI THÍCH THUẬT TOÁN ID3:\n"
        analysis_text += f"{'=' * 50}\n"
        analysis_text += f"1. Tính Entropy của tập dữ liệu gốc\n"
        analysis_text += f"2. Với mỗi thuộc tính:\n"
        analysis_text += f"   - Chia dữ liệu theo giá trị của thuộc tính\n"
        analysis_text += f"   - Tính Entropy có trọng số sau khi chia\n"
        analysis_text += f"   - Tính Information Gain = Entropy_gốc - Entropy_có_trọng_số\n"
        analysis_text += f"3. Chọn thuộc tính có Information Gain cao nhất\n"
        analysis_text += f"4. Tạo nút và lặp lại quá trình cho từng nhánh con\n"

        self.id3_analysis_text.setText(analysis_text)

    def plot_information_gain(self):
        """Vẽ biểu đồ Information Gain"""
        if not self._id3_analysis:
            return

        try:
            self.gain_canvas.figure.clear()
            ax = self.gain_canvas.figure.add_subplot(111)

            # Lấy dữ liệu gain
            analysis = self._id3_analysis[0]  # Root node analysis
            feature_gains = analysis['feature_gains']

            # Sắp xếp theo gain giảm dần
            sorted_items = sorted(feature_gains.items(), key=lambda x: x[1], reverse=True)
            features = [item[0] for item in sorted_items]
            gains = [item[1] for item in sorted_items]

            # Tạo màu sắc (feature được chọn có màu khác)
            colors = ['red' if i == 0 else 'skyblue' for i in range(len(features))]

            # Vẽ biểu đồ
            bars = ax.bar(range(len(features)), gains, color=colors, alpha=0.7)

            # Thêm giá trị lên đầu cột
            for i, (bar, gain) in enumerate(zip(bars, gains)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{gain:.4f}', ha='center', va='bottom', fontsize=10)

                # Thêm dấu sao cho feature được chọn
                if i == 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                            '⭐', ha='center', va='center', fontsize=20)

            ax.set_xlabel('Features')
            ax.set_ylabel('Information Gain')
            ax.set_title('Information Gain Analysis (ID3 Algorithm)')
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45)
            ax.grid(True, alpha=0.3)

            # Thêm chú thích
            ax.text(0.02, 0.98, 'Cột đỏ: Thuộc tính được chọn\nCột xanh: Thuộc tính khác',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            self.gain_canvas.figure.tight_layout()
            self.gain_canvas.draw()

        except Exception as e:
            print(f"Lỗi vẽ biểu đồ Information Gain: {e}")

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
            info_text = f"📊 THÔNG TIN TỔNG QUAN\n"
            info_text += f"{'=' * 50}\n"
            info_text += f"📈 Kích thước: {self._df_raw.shape[0]} dòng × {self._df_raw.shape[1]} cột\n"
            info_text += f"📋 Các cột: {list(self._df_raw.columns)}\n\n"

            # Missing values
            missing = self._df_raw.isnull().sum()
            info_text += f"❌ GIÁ TRỊ THIẾU\n"
            info_text += f"{'=' * 30}\n"
            if missing.sum() == 0:
                info_text += "✅ Không có giá trị thiếu\n\n"
            else:
                info_text += "⚠️ Có giá trị thiếu:\n"
                for col, count in missing[missing > 0].items():
                    percentage = (count / len(self._df_raw)) * 100
                    info_text += f"  • {col}: {count} ({percentage:.1f}%)\n"
                info_text += "\n"

            # Data types
            info_text += f"🔤 KIỂU DỮ LIỆU\n"
            info_text += f"{'=' * 30}\n"
            for col, dtype in self._df_raw.dtypes.items():
                unique_count = self._df_raw[col].nunique()
                info_text += f"  • {col}: {dtype} ({unique_count} giá trị duy nhất)\n"

            # Target analysis if selected
            target_col = self.target_combo.currentText()
            if target_col and target_col in self._df_raw.columns:
                info_text += f"\n🎯 PHÂN TÍCH TARGET: {target_col}\n"
                info_text += f"{'=' * 40}\n"
                target_counts = self._df_raw[target_col].value_counts()
                for value, count in target_counts.items():
                    percentage = (count / len(self._df_raw)) * 100
                    info_text += f"  • {value}: {count} ({percentage:.1f}%)\n"

            self.data_info_text.setText(info_text)

            # Plot target distribution if target is selected
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
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))

            bars = ax.bar(range(len(value_counts)), value_counts.values, color=colors)

            # Thêm giá trị lên đầu cột
            for bar, count in zip(bars, value_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f'{count}', ha='center', va='bottom')

            ax.set_xlabel(target_col)
            ax.set_ylabel('Số lượng')
            ax.set_title(f'Phân phối {target_col}')
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45)
            ax.grid(True, alpha=0.3)

            self.dist_canvas.figure.tight_layout()
            self.dist_canvas.draw()

        except Exception as e:
            print(f"Lỗi vẽ biểu đồ: {e}")

    def train_model(self):
        """Huấn luyện mô hình với phân tích ID3"""
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

            # THỰC HIỆN PHÂN TÍCH ID3 TRƯỚC KHI TRAIN MODEL
            print("Đang phân tích quá trình ID3...")
            self.analyze_id3_process(X, y)
            self.display_id3_analysis()
            self.plot_information_gain()

            # Split data
            test_size = self.test_size_spin.value()
            random_state = self.random_state_spin.value()

            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Train model - FORCE entropy for ID3
            criterion = "entropy"  # Always use entropy for ID3
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
            results_text = f"🎯 KẾT QUẢ HUẤN LUYỆN MÔ HÌNH\n"
            results_text += f"{'=' * 50}\n"
            results_text += f"🔧 Tham số:\n"
            results_text += f"   • Thuật toán: ID3 (Information Gain + Entropy)\n"
            results_text += f"   • Criterion: {criterion}\n"
            results_text += f"   • Max depth: {max_depth}\n"
            results_text += f"   • Test size: {test_size}\n"
            results_text += f"   • Random state: {random_state}\n\n"

            results_text += f"📊 Hiệu suất:\n"
            results_text += f"   • Training accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)\n"
            results_text += f"   • Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)\n"

            # Đánh giá overfitting
            diff = train_acc - test_acc
            if diff > 0.1:
                results_text += f"   ⚠️  Có dấu hiệu overfitting (chênh lệch: {diff:.3f})\n"
            elif diff < 0.05:
                results_text += f"   ✅ Mô hình cân bằng tốt\n"

            results_text += f"\n📈 CLASSIFICATION REPORT\n"
            results_text += f"{'=' * 50}\n"
            results_text += metrics.classification_report(self._y_test, test_pred)

            # Thêm thông tin về cấu trúc cây
            results_text += f"\n🌳 THÔNG TIN CẤU TRÚC CÂY\n"
            results_text += f"{'=' * 40}\n"
            results_text += f"   • Số lượng nút: {self._model.tree_.node_count}\n"
            results_text += f"   • Số lượng lá: {self._model.tree_.n_leaves}\n"
            results_text += f"   • Độ sâu thực tế: {self._model.tree_.max_depth}\n"

            self.results_text.setText(results_text)

            # Plot visualizations - SỬA PHẦN NÀY
            if self.detailed_tree_check.isChecked():
                self.plot_tree_detailed()
            else:
                self.plot_tree()

            self.plot_feature_importance()

            # Setup prediction inputs
            self.setup_prediction_inputs()

            # Chuyển sang tab ID3 Analysis
            self.text_tabs.setCurrentIndex(1)

            QtWidgets.QMessageBox.information(
                self, "Thành công",
                f"✅ Mô hình đã được huấn luyện!\n"
                f"📊 Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)\n"
                f"🔍 Xem tab 'Phân tích ID3' để hiểu quá trình lựa chọn thuộc tính"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Lỗi huấn luyện: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def plot_tree(self):
        """Vẽ cây quyết định với nhãn chi tiết"""
        if self._model is None:
            return

        try:
            self.tree_canvas.figure.clear()
            ax = self.tree_canvas.figure.add_subplot(111)

            # Tạo nhãn tùy chỉnh cho các nút
            feature_names_labeled = []
            for feature in self._feature_names:
                feature_names_labeled.append(f"Feature: {feature}")

            # Tạo nhãn cho các lớp
            class_names_labeled = []
            for class_name in self._model.classes_:
                class_names_labeled.append(f"Class: {str(class_name)}")

            # Vẽ cây với các tham số tùy chỉnh
            plot_tree(self._model,
                      feature_names=feature_names_labeled,
                      class_names=class_names_labeled,
                      filled=True,
                      rounded=True,
                      fontsize=9,
                      ax=ax,
                      impurity=True,  # Hiển thị entropy/gini
                      proportion=False,  # Hiển thị số mẫu thực tế thay vì tỷ lệ
                      precision=3  # Độ chính xác 3 chữ số thập phân
                      )

            # Tùy chỉnh tiêu đề với thông tin thêm
            tree_info = f"Decision Tree (ID3 Algorithm)\n"
            tree_info += f"Criterion: {self._model.criterion} | "
            tree_info += f"Max Depth: {self._model.max_depth} | "
            tree_info += f"Nodes: {self._model.tree_.node_count} | "
            tree_info += f"Leaves: {self._model.tree_.n_leaves}"

            ax.set_title(tree_info, fontsize=12, fontweight='bold', pad=20)

            # Thêm chú thích cho các thành phần của nút
            legend_text = """
    Chú thích các thành phần trong nút:
    • Feature: [Tên thuộc tính] <= [Ngưỡng]: Điều kiện phân chia
    • entropy/gini: Độ đo tạp chất (entropy cho ID3)
    • samples: Số lượng mẫu tại nút này
    • value: [Số mẫu lớp 0, Số mẫu lớp 1, ...]: Phân phối các lớp
    • class: Lớp dự đoán tại nút này (lớp có nhiều mẫu nhất)

    Màu sắc: Càng đậm = Độ tinh khiết càng cao (entropy càng thấp)
    Nút lá: Không có điều kiện phân chia, chỉ có kết quả phân lớp
            """

            # Thêm text box chú thích ở góc dưới trái
            ax.text(0.02, 0.02, legend_text.strip(),
                    transform=ax.transAxes,
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='lightblue',
                              alpha=0.8,
                              edgecolor='navy'))

            self.tree_canvas.figure.tight_layout()
            self.tree_canvas.draw()

        except Exception as e:
            print(f"Lỗi vẽ cây: {e}")

    def plot_tree_detailed(self):
        """Vẽ cây quyết định với thông tin chi tiết hơn (phiên bản mở rộng)"""
        if self._model is None:
            return

        try:
            self.tree_canvas.figure.clear()

            # Tạo subplot với tỷ lệ phù hợp
            fig = self.tree_canvas.figure
            gs = fig.add_gridspec(3, 1, height_ratios=[0.1, 2.5, 0.4], hspace=0.3)

            # Subplot cho tiêu đề
            ax_title = fig.add_subplot(gs[0])
            ax_title.axis('off')

            # Subplot chính cho cây
            ax_main = fig.add_subplot(gs[1])

            # Subplot cho chú thích
            ax_legend = fig.add_subplot(gs[2])
            ax_legend.axis('off')

            # Vẽ cây quyết định
            plot_tree(self._model,
                      feature_names=[f"📊 {feature}" for feature in self._feature_names],
                      class_names=[f"🎯 {str(c)}" for c in self._model.classes_],
                      filled=True,
                      rounded=True,
                      fontsize=8,
                      ax=ax_main,
                      impurity=True,
                      proportion=False,
                      precision=3)

            # Tiêu đề chi tiết
            title_text = f"🌳 DECISION TREE - ID3 ALGORITHM\n"
            title_text += f"Criterion: {self._model.criterion.upper()} | "
            title_text += f"Max Depth: {self._model.max_depth} | "
            title_text += f"Total Nodes: {self._model.tree_.node_count} | "
            title_text += f"Leaf Nodes: {self._model.tree_.n_leaves}"

            ax_title.text(0.5, 0.5, title_text,
                          ha='center', va='center',
                          fontsize=12, fontweight='bold',
                          transform=ax_title.transAxes)

            # Chú thích chi tiết
            legend_content = """
            """

            ax_legend.text(0.02, 0.98, legend_content.strip(),
                           transform=ax_legend.transAxes,
                           verticalalignment='top',
                           horizontalalignment='left',
                           fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.8',
                                     facecolor='lightyellow',
                                     alpha=0.9,
                                     edgecolor='orange',
                                     linewidth=2))

            self.tree_canvas.draw()

        except Exception as e:
            print(f"Lỗi vẽ cây chi tiết: {e}")
            # Fallback về phương thức cũ
            self.plot_tree()

    def plot_feature_importance(self):
        """Vẽ độ quan trọng của features"""
        if self._model is None:
            return

        try:
            self.importance_canvas.figure.clear()
            ax = self.importance_canvas.figure.add_subplot(111)

            importance = self._model.feature_importances_
            indices = np.argsort(importance)[::-1]

            colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
            bars = ax.bar(range(len(importance)), importance[indices], color=colors)

            # Thêm giá trị lên đầu cột
            for bar, imp in zip(bars, importance[indices]):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10)

            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importance (từ mô hình đã huấn luyện)')
            ax.set_xticks(range(len(importance)))
            ax.set_xticklabels([self._feature_names[i] for i in indices], rotation=45)
            ax.grid(True, alpha=0.3)

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

            # Format kết quả hiển thị
            result_text = f"🎯 KẾT QUẢ DỰ ĐOÁN\n"
            result_text += f"{'=' * 30}\n\n"
            result_text += f"🏆 Dự đoán: {prediction}\n\n"
            result_text += f"📊 Xác suất cho từng lớp:\n"

            # Sắp xếp theo xác suất giảm dần
            prob_pairs = list(zip(self._model.classes_, probabilities))
            prob_pairs.sort(key=lambda x: x[1], reverse=True)

            for class_name, prob in prob_pairs:
                emoji = "🥇" if prob == max(probabilities) else "📈"
                result_text += f"   {emoji} {class_name}: {prob:.4f} ({prob * 100:.2f}%)\n"

            # Thêm thông tin input đã nhập
            result_text += f"\n📝 DỮ LIỆU ĐẦU VÀO:\n"
            result_text += f"{'=' * 25}\n"
            for feature, widget in self.pred_inputs.items():
                if isinstance(widget, QtWidgets.QComboBox):
                    value = widget.currentText()
                else:
                    value = widget.text()
                result_text += f"   • {feature}: {value}\n"

            self.pred_result.setText(result_text)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Lỗi dự đoán: {str(e)}")
            self.pred_result.setText("❌ Có lỗi xảy ra trong quá trình dự đoán")


class PlotCanvas(FigureCanvas):
    """Canvas để hiển thị matplotlib plots"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)


# =============================== MAIN ===============================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = ID3AnalyzerGUI()
    gui.show()
    sys.exit(app.exec_())