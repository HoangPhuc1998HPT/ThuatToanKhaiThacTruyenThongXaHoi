import sys
import pandas as pd

from PyQt5 import QtCore,  QtWidgets

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError as e:
    QtWidgets.QMessageBox.critical(None, "Missing dependency", "Bạn cần cài mlxtend: pip install mlxtend")
    raise


class FrequentSetGUI(QtWidgets.QMainWindow):
    """Ứng dụng PyQt5 đơn giản để khai phá luật kết hợp """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Frequent Set Explorer")
        self.resize(900, 450)

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

        file_layout.addWidget(self.fileLine)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        # 2 --- Thông số Apriori --------------------------------------------
        param_layout = QtWidgets.QHBoxLayout()

        self.supportSpin = QtWidgets.QDoubleSpinBox()
        self.supportSpin.setDecimals(3)
        self.supportSpin.setRange(0.0, 1.0)
        self.supportSpin.setSingleStep(0.05)
        self.supportSpin.setValue(0.5)

        self.confSpin = QtWidgets.QDoubleSpinBox()
        self.confSpin.setDecimals(3)
        self.confSpin.setRange(0.0, 1.0)
        self.confSpin.setSingleStep(0.05)
        self.confSpin.setValue(0.7)

        param_layout.addWidget(QtWidgets.QLabel("Min‑Support"))
        param_layout.addWidget(self.supportSpin)
        param_layout.addSpacing(30)
        param_layout.addWidget(QtWidgets.QLabel("Min‑Confidence"))
        param_layout.addWidget(self.confSpin)
        param_layout.addSpacing(50)

        run_btn = QtWidgets.QPushButton("Chạy thuật toán")
        run_btn.clicked.connect(self.run_Freq)
        param_layout.addWidget(run_btn)

        layout.addLayout(param_layout)

        # --- Các vùng hiển thị kết quả ---
        result_layout_label = QtWidgets.QHBoxLayout()
        box_frequent_labels = QtWidgets.QLabel("Tập phổ biến:")
        box_consequent_label = QtWidgets.QLabel("Tập tối đại:")
        box_association_rules_label = QtWidgets.QLabel("Luật kêt hợp tiêu biểu:")
        result_layout_label.addWidget(box_frequent_labels)
        result_layout_label.addWidget(box_consequent_label)
        result_layout_label.addWidget(box_association_rules_label)

        result_layout = QtWidgets.QHBoxLayout()
        self.box_frequent = QtWidgets.QTextEdit(); self.box_frequent.setReadOnly(True)
        self.box_consequent = QtWidgets.QTextEdit(); self.box_consequent.setReadOnly(True)
        self.box_association_rules = QtWidgets.QTextEdit(); self.box_association_rules.setReadOnly(True)
        result_layout.addWidget(self.box_frequent)
        result_layout.addWidget(self.box_consequent)
        result_layout.addWidget(self.box_association_rules)

        layout.addLayout(result_layout_label)
        layout.addLayout(result_layout)

        # Internal data holder
        self._df_raw = None  # DataFrame gốc đọc từ CSV

    # ------------------------------------------------------------------
    #                       SLOT: Browse file
    # ------------------------------------------------------------------
    def browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Chọn file CSV",
            "",
            "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            # Giả định định dạng: cột 0 = Transaction, cột 1 = Item, phân tách ‘;’
            self._df_raw = pd.read_csv(path, sep=';', header=None, names=['Transaction', 'Item'])
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Lỗi đọc file", str(exc))
            return

        self.fileLine.setText(path)

    # ------------------------------------------------------------------
    #                       SLOT: Run Freq
    # ------------------------------------------------------------------
    def run_Freq(self):
        if self._df_raw is None:
            QtWidgets.QMessageBox.warning(self, "Chưa có dữ liệu", "Vui lòng chọn file CSV trước.")
            return

        min_sup = self.supportSpin.value()
        min_conf = self.confSpin.value()

        # Gom thành list giao dịch
        transactions = (
            self._df_raw.groupby('Transaction')['Item']
            .apply(list)
            .tolist()
        )

        # Encoding 1‑0 (Encode dữ liệu)
        te = TransactionEncoder()
        te_arr = te.fit(transactions).transform(transactions)
        df_freq = pd.DataFrame(te_arr, columns=te.columns_)

        # Apriori
        freq = apriori(df_freq, min_support=min_sup, use_colnames=True)
        freq.sort_values(by='support', ascending=False, inplace=True)
        freq_meg = f"Các tập phổ biến có Min-Support = {min_sup:.3f} : \n"
        freq_meg += f"\n{freq}"

        self.box_frequent.setText(freq_meg)

        # Tìm tập phổ biến tối đại:
        freq['length'] = freq['itemsets'].apply(lambda x: len(x))
        maximal_itemsets=[]
        for i, row in freq.iterrows():
            if not any(row['itemsets'] < other['itemsets'] for _, other in freq.iterrows()):
                maximal_itemsets.append(row)

        max_freq_meg = "Các tập phổ biến tối đại: \n"
        for item in maximal_itemsets:
            max_freq_meg += f"\n{item['itemsets']} -> support   {item['support']:.2f}"

        self.box_consequent.setText(max_freq_meg)

        # Luật kết hợp
        rules = association_rules(freq, metric='confidence', min_threshold=min_conf)
        rules.sort_values('confidence', ascending=False, inplace=True)
        # Chọn 3 luật ví dụ để in
        rules_meg = f"5 luật kết hợp thỏa Min-Confidence =  {min_conf:.3f} :\n"
        for i, row in rules.sample(5).iterrows():
            rules_meg += f"\n{set(row['antecedents'])} => {set(row['consequents'])} | confidence = {row['confidence']:.2f}"

        self.box_association_rules.setText(rules_meg)

# =============================== MAIN ===============================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = FrequentSetGUI()
    gui.show()
    sys.exit(app.exec_())
