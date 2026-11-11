"""Styled Excel output (openpyxl): table(), text_sheet(), write(path, readme, sheets), add_sheets(path, sheets)."""
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

DARK, LIGHT, ZEBRA = "1F4E5F", "D9E7EC", "F5F9FA"
HEAD = Font(bold=True, color="FFFFFF")
TITLE = Font(bold=True, size=14, color=DARK)
LINE = Border(bottom=Side(style="thin", color="BFBFBF"))


# ------------------------------------------------------------------ one table or text sheet, styled
def table(ws, df, row=1, title=None, note=None):
    """Write df at `row` with a title, coloured header, zebra rows, filters and sensible widths."""
    if title:
        ws.cell(row, 1, title).font = TITLE
        row += 1
    if note:
        ws.cell(row, 1, note).font = Font(italic=True, color="595959")
        row += 1
    df = df.copy() if len(df.columns) else pd.DataFrame({"(empty)": []})
    for c in df.columns:                                   # Excel can't store tz-aware / timedelta
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].map({True: "yes", False: ""})
    for j, col in enumerate(df.columns, 1):
        cell = ws.cell(row, j, str(col))
        cell.font, cell.fill = HEAD, PatternFill("solid", fgColor=DARK)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for i, values in enumerate(df.itertuples(index=False), row + 1):
        for j, v in enumerate(values, 1):
            cell = ws.cell(i, j, None if pd.isna(v) else v)
            cell.border = LINE
            if (i - row) % 2 == 0:
                cell.fill = PatternFill("solid", fgColor=ZEBRA)
            if isinstance(v, float):
                cell.number_format = "0.0"
            elif isinstance(v, pd.Timestamp):
                cell.number_format = "yyyy-mm-dd hh:mm" if (v.hour or v.minute) else "yyyy-mm-dd"
    for j, col in enumerate(df.columns, 1):
        width = max([len(str(col))] + [len(str(v)) for v in df.iloc[:200, j - 1]])
        dim = ws.column_dimensions[get_column_letter(j)]
        dim.width = max(dim.width or 0, min(max(width + 2, 8), 45))      # several tables share columns
    if row == 1:
        ws.freeze_panes = "B2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(df.columns))}{len(df) + 1}"
    return row + len(df) + 2


def text_sheet(ws, title, lines):
    """Sheet with a title and lines of text (the read me)."""
    ws.cell(1, 1, title).font = TITLE
    for i, line in enumerate(lines, 3):
        c = ws.cell(i, 1, line)
        if line and not line.startswith(" "):
            c.font = Font(bold=True, color=DARK)
    ws.column_dimensions["A"].width = 130


# ------------------------------------------------------------------ workbooks: write a new one, add sheets to an existing one
def write(path, readme, sheets):
    """sheets: list of (sheet name, [(title, note, df), ...])."""
    wb = Workbook()
    if readme:
        text_sheet(wb.active, "HEATWATCH datasets", readme)
        wb.active.title = "Read me"
    else:
        wb.remove(wb.active)
    for name, blocks in sheets:
        add_sheet(wb, name, blocks)
    return save(wb, path)


def add_sheet(wb, name, blocks):
    """One sheet with the tables in blocks [(title, note, df), ...]; an existing sheet of that name is replaced."""
    if name[:31] in wb.sheetnames:
        wb.remove(wb[name[:31]])
    ws = wb.create_sheet(name[:31])
    row = 1
    for title, note, df in blocks:
        row = table(ws, df, row, title, note)
    ws.sheet_properties.tabColor = LIGHT if name[:1] in "AB" else DARK
    return ws


def add_sheets(path, sheets):
    """Add sheets (list of (name, blocks)) to an existing workbook; sheets with the same name are replaced."""
    wb = load_workbook(path)
    for name, blocks in sheets:
        add_sheet(wb, name, blocks)
    return save(wb, path)


def save(wb, path):
    """Save; if the file is open in Excel, save a copy next to it."""
    try:
        wb.save(path)
    except PermissionError:                # file is open in Excel: save a copy next to it
        path = path.with_name(f"{path.stem}_{pd.Timestamp.now():%H%M}{path.suffix}")
        wb.save(path)
        print(f"  ! {path.name}: original is open in Excel, saved as this copy instead")
    return path
