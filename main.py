import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from config import APP_NAME, ORGANIZATION_NAME
from gui import ImageMatchingWindow


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)

    window = ImageMatchingWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
