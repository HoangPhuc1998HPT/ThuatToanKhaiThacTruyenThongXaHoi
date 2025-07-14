import pandas as pd
from itertools import combinations
from PyQt5 import QtWidgets, QtCore
import sys

# =======================
# HÀM XỬ LÝ DỮ LIỆU
# =======================
def is_reduct(df, subset, decision_attr):
    grouped = df.groupby(subset)[decision_attr].apply(set)
    return all(len(g) == 1 for g in grouped)

def find_minimal_reducts(df, attrs, decision_attr):
    valid = []
    for r in range(1, len(attrs) + 1):
        for combo in combinations(attrs, r):
            grouped = df.groupby(list(combo))[decision_attr].apply(set)
            if all(len(g) == 1 for g in grouped):
                if not any(set(smaller).issubset(combo) for smaller in valid):
                    valid = [r for r in valid if not set(combo).issubset(set(r))]
                    valid.append(combo)
    return valid

def dependency_coefficient(df, B_attrs, decision_attr):
    #Tìm tập IND(B)  bằng cách nhóm các đối tượng theo tập điều kiện
    group = df.groupby(B_attrs)[decision_attr].apply(set)
    #lower_total = sum(len(df.groupby(B_attrs).get_group(key)) for key, values in group.items() if len(values) == 1)
    lower_total = 0
    # Lặp qua từng khóa và giá trị trong 'group.items()'
    # key là giá trị trong tập IND(B), ứng với 1 key có thể chứa nhiều đối tượng
    #values là giá trị quyết định trong tập C tương ứng với key
    for key, values in group.items():
        # Kiểm tra xem độ dài của 'values' có bằng 1 nghĩa là chỉ có 1 quyết định cho 1 giá trị trong IND(B)
        # Nếu key > 1 nghĩa là có nhiều hớn quyết định cho cùng 1 tập điều kiện thì giá trị đó là thô
        # thuộc vùng biên, không ảnh hưởng đến hụ thuộc nên có thể bỏ qua
        if len(values) == 1:
            # Nếu đúng, lấy nhóm từ DataFrame 'df' dựa trên khóa 'key'
            # và thuộc tính 'B_attrs', sau đó tính độ dài của nhóm đó
            group_length = len(df.groupby(B_attrs).get_group(key))
            # Cộng độ dài này vào 'lower_total'
            lower_total += group_length
    return lower_total / len(df)

def approximation(df, X, B):
    grouped = df.groupby(B)[df.columns[0]].apply(lambda g: set(g.astype(str)))
    lower = set()
    upper = set()
    for group in grouped:
        if group <= X:
            lower |= group
        if group & X:
            upper |= group
    return lower, upper

# =======================
# GIAO DIỆN CHÍNH
# =======================
class ReductFinderApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rough Set Explorer")
        self.resize(1200, 750)

        self.df = None
        self.id_col = None
        self.condition_cols = []
        self.decision_col = None
        self.X_set = set()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # Load file và tập quyết định C
        file_layout = QtWidgets.QHBoxLayout()
        self.fileLine = QtWidgets.QLineEdit()
        self.fileLine.setReadOnly(True)
        self.decisionLabel = QtWidgets.QLabel("Tập quyết định C = ")
        self.load_btn = QtWidgets.QPushButton("Chọn file CSV")
        self.load_btn.clicked.connect(self.load_csv)

        file_layout.addWidget(self.fileLine)
        file_layout.addWidget(self.load_btn)
        spacerItem_1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        file_layout.addItem(spacerItem_1)
        file_layout.addWidget(self.decisionLabel)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        file_layout.addSpacerItem(spacerItem)

        main_layout.addLayout(file_layout)

        # Layout cho các nhóm chức năng
        top_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_layout)

        # --- Nhóm chọn tập B ---
        self.attr_group = QtWidgets.QGroupBox("Chọn tập thuộc tính B")
        self.attr_layout = QtWidgets.QGridLayout()
        self.attr_group.setLayout(self.attr_layout)
        top_layout.addWidget(self.attr_group)

        # --- Nhóm tập X ---
        self.X_group = QtWidgets.QGroupBox("Tập X")
        X_layout = QtWidgets.QVBoxLayout()

        X_select_layout = QtWidgets.QHBoxLayout()
        self.X_combo = QtWidgets.QComboBox()
        self.X_add_btn = QtWidgets.QPushButton("+ Thêm")
        self.X_remove_btn = QtWidgets.QPushButton("- Xóa")

        self.X_add_btn.clicked.connect(self.add_to_X)
        self.X_remove_btn.clicked.connect(self.remove_from_X)
        X_select_layout.addWidget(self.X_combo)
        X_select_layout.addWidget(self.X_add_btn)
        X_select_layout.addWidget(self.X_remove_btn)

        X_select_all_layout = QtWidgets.QHBoxLayout()
        self.X_select_accept_btn = QtWidgets.QPushButton("Chọn X theo giá trị quyết định")
        self.X_reset_btn = QtWidgets.QPushButton("Reset X")

        self.X_select_accept_btn.clicked.connect(self.select_X_by_class)
        self.X_reset_btn.clicked.connect(self.reset_X)

        X_select_all_layout.addWidget(self.X_select_accept_btn)
        X_select_all_layout.addWidget(self.X_reset_btn)

        self.X_label = QtWidgets.QLabel("Tập X = o")

        X_layout.addLayout(X_select_layout)
        X_layout.addLayout(X_select_all_layout)
        X_layout.addWidget(self.X_label)
        self.X_group.setLayout(X_layout)
        top_layout.addWidget(self.X_group)

        # --- Các nút chức năng ---
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_dependency = QtWidgets.QPushButton("Tìm Phụ thuộc C vào B")
        self.btn_approx = QtWidgets.QPushButton("Tìm Xấp xỉ tập X qua B")
        self.btn_reduct = QtWidgets.QPushButton("Tìm reduct rút gọn")
        self.btn_dependency.clicked.connect(self.check_dependency)
        self.btn_approx.clicked.connect(self.compute_approximation)
        self.btn_reduct.clicked.connect(self.find_reducts)
        button_layout.addWidget(self.btn_dependency)
        button_layout.addWidget(self.btn_approx)
        button_layout.addWidget(self.btn_reduct)
        main_layout.addLayout(button_layout)

        # --- Các vùng hiển thị kết quả ---
        result_layout = QtWidgets.QHBoxLayout()
        self.box_dependency = QtWidgets.QTextEdit(); self.box_dependency.setReadOnly(True)
        self.box_approx = QtWidgets.QTextEdit(); self.box_approx.setReadOnly(True)
        self.box_reduct = QtWidgets.QTextEdit(); self.box_reduct.setReadOnly(True)
        result_layout.addWidget(self.box_dependency)
        result_layout.addWidget(self.box_approx)
        result_layout.addWidget(self.box_reduct)
        main_layout.addLayout(result_layout)

    def load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Chọn file dữ liệu", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path, sep=';')
            if df.shape[1] < 3:
                return
            self.df = df
            self.id_col = df.columns[0]
            self.condition_cols = df.columns[1:-1].tolist()
            self.decision_col = df.columns[-1]
            self.X_set = set()
            self.X_label.setText("Tập X = ∅")

            for i in reversed(range(self.attr_layout.count())):
                widget = self.attr_layout.itemAt(i).widget()
                if widget: widget.setParent(None)
            for idx, col in enumerate(self.condition_cols):
                cb = QtWidgets.QCheckBox(col)
                self.attr_layout.addWidget(cb, idx//4, idx%4)

            self.X_combo.clear()
            self.X_combo.addItems(df[self.id_col].astype(str).tolist())

            self.fileLine.setText(path)
            self.decisionLabel.setText(f"Tập quyết định C = {self.decision_col}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi đọc file", str(e))

    def get_selected_B(self):
        return [self.attr_layout.itemAt(i).widget().text()
                for i in range(self.attr_layout.count())
                if self.attr_layout.itemAt(i).widget().isChecked()]

    def add_to_X(self):
        val = self.X_combo.currentText()
        if val: self.X_set.add(val)
        self.update_X_label()

    def remove_from_X(self):
        val = self.X_combo.currentText()
        self.X_set.discard(val)
        self.update_X_label()

    def reset_X(self):
        self.X_set.clear()
        self.last_selected_condition = ""
        self.update_X_label()

    def update_X_label(self):
        n = len(self.X_set)
        if n == 0:
            s = "o"
        elif n <= 15:
            s = ", ".join(sorted(self.X_set))
        else:
            firsts = ", ".join(sorted(self.X_set)[:3])
            lasts = ", ".join(sorted(self.X_set)[-3:])
            s = f"{firsts}, …, {lasts} ({n} phần tử)"

        # Thêm điều kiện nếu chọn từ lớp quyết định
        if hasattr(self, 'last_selected_condition') and self.last_selected_condition:
            suffix = f" thỏa điều kiện: \"{self.last_selected_condition}\""
        else:
            suffix = ""

        self.X_label.setText(f"Tập X = {{{s}}}{suffix}")

    def select_X_by_class(self):
        self.X_set.clear()
        if self.df is None:
            return
        values = self.df[self.decision_col].dropna().unique().tolist()
        value, ok = QtWidgets.QInputDialog.getItem(self, "Chọn lớp quyết định", "Lớp:", values, 0, False)
        if ok:
            self.last_selected_condition = value  # lưu lại điều kiện quyết định được chọn
            self.X_set = set(self.df[self.df[self.decision_col] == value][self.id_col].astype(str).tolist())
            self.update_X_label()

    def check_dependency(self):
        try:
            if self.df is None: return
            B = self.get_selected_B()
            if not B:
                self.box_dependency.setText("Chú ý: Chưa chọn tập B")
                return
            k = dependency_coefficient(self.df, B, self.decision_col)
            msg = f"Tập B: {B}\nQuyết định C: {self.decision_col}\nHệ số k = {k:.2f}\n\n"
            msg += "Kết luận: C CÓ phụ thuộc hoàn toàn vào B" if k == 1.0 else "Kết luận: C KHÔNG phụ thuộc hoàn toàn vào B"
            self.box_dependency.setText(msg)
        except Exception as e:
            self.box_dependency.setText(f"Có Lỗi: {e}")

    def compute_approximation(self):
        try:
            if self.df is None or not self.X_set:
                self.box_approx.setText("Chú ý: Chưa chọn tập X hoặc tập B")
                return
            B = self.get_selected_B()
            if not B:
                self.box_approx.setText("Chú ý: Chưa chọn tập B")
                return
            lower, upper = approximation(self.df, self.X_set, B)
            a = len(lower)/len(upper) if upper else 0
            msg = f"Hệ số α = {len(lower)}/{len(upper)} = {a:.2f}\n\n"
            msg += "Chi tiết: \n"
            msg += f"Tập B: {B}\nTập X: {sorted(self.X_set)}\n\n"
            msg += f"Lower(B,X): {sorted(lower)}\nUpper(B,X): {sorted(upper)}\n"
            #msg += f"Hệ số α = {len(lower)}/{len(upper)} = {a:.2f}"
            self.box_approx.setText(msg)
        except Exception as e:
            self.box_approx.setText(f"Có Lỗi: {e}")

    def find_reducts(self):
        try:
            if self.df is None:
                self.box_reduct.setText("Chú ý: Chưa có dữ liệu")
                return
            reducts = find_minimal_reducts(self.df, self.condition_cols, self.decision_col)
            if reducts:
                msg = "Tìm thấy Reducts tối thiểu:\n\n" +'{'+'\n{'.join(str(r) +'}' for r in reducts)
            else:
                msg = "Không tìm được reduct nào."
            self.box_reduct.setText(msg)
        except Exception as e:
            self.box_reduct.setText(f"Có Lỗi: {e}")

# =======================
# CHẠY ỨNG DỤNG
# =======================
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = ReductFinderApp()
    win.show()
    sys.exit(app.exec_())

