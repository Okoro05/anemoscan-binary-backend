"""Tests for the AnemoScan PDF report brand asset."""

import io

try:
    from api.app.admin_routes import ReportLogo
except ModuleNotFoundError:
    from app.admin_routes import ReportLogo

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate


def test_anemoscan_svg_logo_renders_in_pdf_report():
    output = io.BytesIO()
    document = SimpleDocTemplate(output, pagesize=A4)
    logo = ReportLogo()

    assert logo.width > 0
    assert logo.height > 0

    document.build([logo])
    assert output.getvalue().startswith(b"%PDF")
