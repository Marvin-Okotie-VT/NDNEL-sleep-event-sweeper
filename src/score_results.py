"""Write scoring results to Excel 
layout:
Time | Elapsed Time | Score | Unsure_Score | Notes  (one row per 30-s epoch)
A second sheet, "Run info", records the file, settings and code version that produced it.
"""
from datetime import datetime, timedelta

from openpyxl import Workbook
from openpyxl.utils import get_column_letter

HEADERS = ["Time", "Elapsed Time", "Score", "Unsure_Score", "Notes"]
WIDTHS = [12.2, 25.7, 9, 12.8, 90]


def write_scores(path, start: datetime, results, run_info: dict):
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(HEADERS)
    for r in results:
        clock = (start + timedelta(seconds=r.start_s)).time()    
        elapsed = timedelta(seconds=r.start_s)
        ws.append([clock, elapsed, r.score or None, r.unsure or None, r.notes or None])
    for row in ws.iter_rows(min_row=2):
        row[0].number_format = "h:mm:ss"
        row[1].number_format = "[h]:mm:ss"
    for i, w in enumerate(WIDTHS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"

    info = wb.create_sheet("Run info")
    for k, v in run_info.items():
        info.append([k, str(v)])
    info.column_dimensions["A"].width = 28
    info.column_dimensions["B"].width = 100
    wb.save(path)