
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QVBoxLayout, QPushButton, QLabel

from Frontend.ButtonUI import ButtonUI
from Frontend.GlobalStyle import GlobalStyle
from Rough_Set_Analysis.rough_set_analysis import ReductFinderApp


class LandlordMenu(QWidget):
    def __init__(self, main_window=None, user_id=None):
        super().__init__()
        self.setStyleSheet(GlobalStyle.global_stylesheet())
        print("[DEBUG] LandlordMenu khởi tạo")

        self.main_window = main_window
        self.current_page = None


        self.main_window.setWindowTitle("Dashboard Chủ trọ")
        #self.main_window.setGeometry(300, 100, 1000, 600)
        self.main_window.resize(GlobalStyle.WINDOW_WIDTH, GlobalStyle.WINDOW_HEIGHT)
        self.main_window.setMinimumSize(GlobalStyle.WINDOW_WIDTH, GlobalStyle.WINDOW_HEIGHT)
        self.main_window.setMaximumSize(GlobalStyle.WINDOW_WIDTH, GlobalStyle.WINDOW_HEIGHT)

        #self.main_window.setStyleSheet("""
            #background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #FF6B6B, stop:1 #FFA07A);
           # border-radius: 15px;
        #""")
        self.main_layout = QHBoxLayout()
       #self.main_layout.setContentsMargins(0, 0, 0, 0)

        # ------------ LEFT MENU FRAME ------------
        self.left_frame = QWidget()
        self.left_frame.setFixedWidth(250)
        self.left_frame.setStyleSheet("""
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #FF6B6B, stop:1 #FFA07A);
            border-radius: 15px;
        """)

        left_layout = QVBoxLayout(self.left_frame)
        left_layout.setAlignment(Qt.AlignTop)

        # Label chào mừng
        self.label_landlord = QLabel("👋 Chào mừng đến với DASHBOARD Chủ trọ: Nguyễn Văn A")
        self.label_landlord.setObjectName("Title")

        #self.label_landlord.setStyleSheet("color: white; font-weight: bold; padding: 10px;")
        #left_layout.addWidget(self.label_landlord)

        # Tạo nút và áp dụng style\
        button_ui = ButtonUI.landlord_dashboard_button()

        self.home_btn = QPushButton("Giới thiệu")
        button_ui.apply_style(self.home_btn)
        self.home_btn.clicked.connect(lambda: print("Ở đây giới thiệu về ứng dụng và báo cáo"))

        self.info_btn = QPushButton("Thuật toán rút gọn - Reduct")
        button_ui.apply_style(self.info_btn)
        self.info_btn.clicked.connect(lambda: self.set_right_frame(ReductFinderApp))

        self.infor_list_room_btn = QPushButton("Phân lớp bằng Naïve Bayes")
        button_ui.apply_style(self.infor_list_room_btn)
        self.infor_list_room_btn.clicked.connect(lambda : print("3"))

        self.create_new_room_btn = QPushButton("Phân lớp bằng Cây quyết định")
        button_ui.apply_style(self.create_new_room_btn)
        self.create_new_room_btn.clicked.connect(lambda : print("3"))

        self.infor_list_invoice_btn = QPushButton("Gom cụm - Clustering")
        button_ui.apply_style(self.infor_list_invoice_btn)
        self.infor_list_invoice_btn.clicked.connect(lambda : print("4"))

        self.add_adv_find_tenant_btn = QPushButton("K-means")
        button_ui.apply_style(self.add_adv_find_tenant_btn)
        self.add_adv_find_tenant_btn.clicked.connect(lambda : print("5"))

        self.add_list_maintenance_btn = QPushButton("Minkowski - Euclidean ")
        button_ui.apply_style(self.add_list_maintenance_btn)
        self.add_list_maintenance_btn.clicked.connect(lambda : print("6"))

        self.logout_btn = QPushButton("Manhattan distance")
        button_ui.apply_style(self.logout_btn)
        self.logout_btn.clicked.connect(lambda : print("7"))

        self.exist_btn = QPushButton("❌ Thoát")
        button_ui.apply_style(self.exist_btn)
        self.exist_btn.clicked.connect(lambda: self.close_window_menu())

        # Thêm tất cả các button vào layout
        left_layout.addWidget(self.home_btn)
        left_layout.addWidget(self.info_btn)
        left_layout.addWidget(self.infor_list_room_btn)
        left_layout.addWidget(self.create_new_room_btn)
        left_layout.addWidget(self.add_list_maintenance_btn)
        left_layout.addWidget(self.infor_list_invoice_btn)
        left_layout.addWidget(self.add_adv_find_tenant_btn)
        left_layout.addWidget(self.logout_btn)
        left_layout.addWidget(self.exist_btn)

        # ----------- RIGHT FRAME (QStackedWidget) -----------
        self.right_frame = QWidget()
        self.right_layout = QVBoxLayout(self.right_frame)
        self.right_layout.setContentsMargins(0, 0, 0, 0)



        # Thêm vào layout chính
        self.main_layout.addWidget(self.left_frame)
        self.main_layout.addWidget(self.right_frame)



        # Thành:
        #LandlordController.go_to_home_page(self, self.id_lanlord)

        self.setLayout(self.main_layout)

    def set_right_frame(self, PageClass):
        if self.current_page:
            self.right_layout.removeWidget(self.current_page)
            self.current_page.setParent(None)

        try:
            if callable(PageClass):  # lambda trả về instance
                self.current_page = PageClass()
            else:
                self.current_page = PageClass(self.main_window, self.id_lanlord)
        except TypeError as e:
            print(f"[⚠️ Cảnh báo] {PageClass.__name__} không nhận 2 tham số: {e}")
            self.current_page = PageClass(self.main_window)

        self.right_layout.addWidget(self.current_page)
        return self.current_page



    def close_window_menu(self):
        self.main_window.close()

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    main_window = QWidget()
    menu = LandlordMenu(main_window)
    main_window.setLayout(menu.main_layout)
    main_window.show()
    sys.exit(app.exec_())