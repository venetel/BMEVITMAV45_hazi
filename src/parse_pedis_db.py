"""Extract image-level labels from the legacy PEDIS database (PEDI3.MDB).

PEDI3.MDB is an old Microsoft Jet (1.x/2.0 era) database. Its header page is
blanked and the page layout predates the formats modern tools support, so
neither ODBC/DAO nor mdbtools/access-parser can open it. This module reads the
raw pages directly.

Reverse-engineered layout (2048-byte pages, little endian):

Page header
    [0]       page type: 6 = data page, 5 = table header, 4 = index, ...
    [4:8]     owning table id (data pages)
    [8:10]    number of row slots
    [20:...]  uint16 row offsets; low 11 bits = start inside the page,
              high bits = flags (0x2000 alone marks a 4-byte forwarding
              pointer, 0x3000 marks the forwarded row itself)
    A row ends where the previous slot's row starts (rows grow downwards).

Row
    [0:2]     row length in bytes
    [2]       number of fixed columns (nf)
    [3]       number of variable columns stored in this row
    [4:...]   fixed-size column data, then variable-length data
    tail      ceil(nf / 8) bytes of bit-column flags (last bytes of the row)
              before it: jump table, (len - 1) // 256 bytes; each entry is the
              index of the first variable column whose offset is >= 256 * n
              before it: variable column count
              before it: (count + 1) one-byte offsets, stored in reverse order

The column catalog (MSysColumns) is stored in the same format, so the schema
of the patient table is read from the file itself instead of hard-coding it.

Privacy: the patient table contains names, national health insurance numbers
(TAJ), addresses and phone numbers. None of these are written to the output;
patients are replaced by sequential pseudonymous ids. Never commit PEDI3.MDB
or derived files that contain direct identifiers.
"""

import argparse
import collections
import csv
import os
import re
import struct
import sys
from dataclasses import dataclass

PAGE_SIZE = 2048
PAGE_TYPE_DATA = 6
MSYS_COLUMNS_TABLE_ID = 4
OFFSET_MASK = 0x07FF
FLAG_FORWARD_POINTER = 0x2000
FLAG_FORWARDED_ROW = 0x1000
ENCODING = "cp1250"  # Central European Windows code page (Hungarian text)

# Column type codes found in MSysColumns.
TYPE_BIT, TYPE_BYTE, TYPE_INT, TYPE_LONG = 0x01, 0x02, 0x03, 0x04
TYPE_TEXT, TYPE_MEMO = 0x0A, 0x0C

# Columns that are safe to export. Everything else (name, TAJ number, address,
# phone, doctor, prescription numbers, free-text remarks) is dropped.
EXPORTED_TEXT_COLUMNS = ["kor", "suly", "cmeret", "statusz", "diagn1", "diagn2", "diagn3", "seszk", "dtip"]
EXPORTED_BIT_COLUMNS = ["diabetes", "ferfi", "no", "gyerek"]


@dataclass
class Column:
    name: str
    table_id: int
    index: int        # position among fixed columns, or among variable columns
    size: int
    col_type: int
    is_variable: bool


def iter_rows(db: bytes, table_id: int):
    """Yield the raw bytes of every live row that belongs to `table_id`."""
    for page_no in range(len(db) // PAGE_SIZE):
        page = db[page_no * PAGE_SIZE:(page_no + 1) * PAGE_SIZE]
        if page[0] != PAGE_TYPE_DATA or struct.unpack_from("<I", page, 4)[0] != table_id:
            continue
        n_slots = struct.unpack_from("<H", page, 8)[0]
        slots = struct.unpack_from(f"<{n_slots}H", page, 20)
        end = PAGE_SIZE
        for slot in slots:
            start = slot & OFFSET_MASK
            if start >= end:  # empty / deleted slot
                continue
            row = page[start:end]
            end = start
            flags = slot & ~OFFSET_MASK
            if flags & FLAG_FORWARD_POINTER and not flags & FLAG_FORWARDED_ROW:
                continue  # 4-byte pointer; the row itself is yielded on its own page
            yield row


def decode_row(row: bytes):
    """Split a row into (fixed_bytes, bit_flags, variable_values) or return None if malformed."""
    if len(row) < 8 or struct.unpack_from("<H", row, 0)[0] != len(row):
        return None
    n_fixed = row[2]
    n_bit_bytes = (n_fixed + 7) // 8
    n_jumps = (len(row) - 1) // 256
    tail = len(row) - n_bit_bytes
    jumps = sorted(row[tail - 1 - j] for j in range(n_jumps))
    n_var = row[tail - n_jumps - 1]
    offsets_base = tail - n_jumps - 2
    if n_var != row[3] or offsets_base - n_var < 4 or not n_var:
        return None

    offsets = []
    for k in range(n_var):
        high = sum(1 for jump in jumps if k >= jump)
        offsets.append(row[offsets_base - k] + 256 * high)
    # The end-of-data sentinel is not reliably covered by the jump table (a jump
    # entry equal to n_var may only mean "nothing crosses"), so derive it from
    # the position of the offset table and just check its low byte.
    data_end = offsets_base - n_var
    if row[data_end] != data_end & 0xFF:
        return None
    offsets.append(data_end)
    if offsets[0] < 4 or any(a > b for a, b in zip(offsets, offsets[1:])):
        return None

    values = [row[offsets[k]:offsets[k + 1]] for k in range(n_var)]
    bits = int.from_bytes(row[tail:], "little")
    return row[4:offsets[0]], bits, values


def read_schema(db: bytes):
    """Read MSysColumns and return {table_id: [Column, ...]}."""
    tables = collections.defaultdict(list)
    for row in iter_rows(db, MSYS_COLUMNS_TABLE_ID):
        decoded = decode_row(row)
        if decoded is None or not decoded[2]:
            continue
        fixed, _, values = decoded
        table_id = struct.unpack_from("<I", fixed, 0)[0]
        raw_index, size, col_type = fixed[6], fixed[7], fixed[8]
        is_variable = raw_index >= 0x80
        name = values[0].decode(ENCODING, errors="replace").rstrip("\x00")
        tables[table_id].append(Column(name, table_id, 0xFF - raw_index if is_variable else raw_index,
                                       size, col_type, is_variable))
    return tables


def find_patient_table(schema):
    for table_id, columns in schema.items():
        names = {c.name for c in columns}
        if {"kepaz", "felvdat", "diagn1"} <= names and "nev" in names:
            return table_id, columns
    raise RuntimeError("Patient table (kepaz/felvdat/diagn1) not found in MSysColumns")


def clean_text(raw: bytes) -> str:
    return raw.decode(ENCODING, errors="replace").replace("\x00", "").strip()


def parse_date(text: str) -> str:
    """Convert the yy-mm-dd strings used by PEDIS to ISO dates (empty if invalid)."""
    m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{2})", text)
    if not m:
        return ""
    yy, mm, dd = (int(g) for g in m.groups())
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return ""
    year = 2000 + yy if yy <= 30 else 1900 + yy
    return f"{year:04d}-{mm:02d}-{dd:02d}"


def read_memo(db: bytes, field: bytes) -> bytes:
    """Resolve a memo field: either inline data or a pointer to a long-value row.

    Memo header: uint32 length; bit 31 set = data follows inline. Otherwise
    bytes [4] = row slot and [5:8] = page number of the long-value row, which
    starts with a 10-byte header (uint16 length, uint32 next pointer, 4 bytes).
    """
    if len(field) < 4:
        return b""
    header = struct.unpack_from("<I", field, 0)[0]
    if header & 0x80000000:
        return field[4:]
    length = header & 0x00FFFFFF
    data = b""
    pointer = field[4:8]
    while pointer != b"\x00\x00\x00\x00" and len(data) < length:
        slot_no, page_no = pointer[0], int.from_bytes(pointer[1:4], "little")
        page = db[page_no * PAGE_SIZE:(page_no + 1) * PAGE_SIZE]
        if len(page) < PAGE_SIZE or page[0] != PAGE_TYPE_DATA:
            break
        n_slots = struct.unpack_from("<H", page, 8)[0]
        if slot_no >= n_slots:
            break
        slots = struct.unpack_from(f"<{n_slots}H", page, 20)
        start = slots[slot_no] & OFFSET_MASK
        end = min([s & OFFSET_MASK for s in slots[:slot_no] if s & OFFSET_MASK > start], default=PAGE_SIZE)
        row = page[start:end]
        data += row[10:]
        pointer = row[2:6]
    return data[:length]


def rtf_to_text(rtf: bytes) -> str:
    """Very small RTF-to-plain-text converter, sufficient for the e-record notes."""
    text = rtf.decode("latin-1")
    if not text.startswith("{\\rtf"):
        return rtf.decode(ENCODING, errors="replace").replace("\x00", "").strip()
    # Drop destination groups such as the font and colour tables.
    text = re.sub(r"\{\\(?:fonttbl|colortbl|stylesheet|\*)[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", text)
    text = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: bytes([int(m.group(1), 16)]).decode(ENCODING, errors="replace"), text)
    text = re.sub(r"\\(?:par|line)\b ?", "\n", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace("{", "").replace("}", "").replace("\r", "")
    return re.sub(r"\n\s*\n+", "\n", text).replace("\x00", "").strip()


def normalize_person_key(taj: str, name: str) -> str:
    """Key used to group visits of the same patient (never written out)."""
    digits = re.sub(r"\D", "", taj)
    if len(digits) == 9:
        return "taj:" + digits
    return "name:" + re.sub(r"\s+", " ", name.lower()).strip() if name else ""


def columns_by_name(columns):
    var_index = {c.name: c.index for c in columns if c.is_variable}
    bit_index = {c.name: c.index for c in columns if not c.is_variable and c.col_type == TYPE_BIT}
    return var_index, bit_index


def extract_records(db: bytes):
    """Return (image records, e-record notes, skipped row count) without direct identifiers."""
    schema = read_schema(db)
    table_id, columns = find_patient_table(schema)
    var_index, bit_index = columns_by_name(columns)

    pseudonyms = {}

    def pseudonym(key):
        return pseudonyms.setdefault(key, len(pseudonyms)) if key else -1

    records, skipped = [], 0
    for row in iter_rows(db, table_id):
        decoded = decode_row(row)
        if decoded is None:
            skipped += 1
            continue
        _, bits, values = decoded

        def text(name):
            i = var_index[name]
            return clean_text(values[i]) if i < len(values) else ""

        image_id = text("kepaz").upper()
        if not image_id:
            skipped += 1
            continue
        record = {"image_id": image_id,
                  "patient_id": pseudonym(normalize_person_key(text("tbszam"), text("nev"))),
                  "exam_date": parse_date(text("felvdat"))}
        for name in EXPORTED_TEXT_COLUMNS:
            if name in var_index:
                record[name] = text(name)
        for name in EXPORTED_BIT_COLUMNS:
            if name in bit_index:
                record[name] = (bits >> bit_index[name]) & 1
        records.append(record)

    # Electronic patient record ("ekorlap"): free-text RTF notes keyed by TAJ number.
    notes = []
    for ekorlap_table, ekorlap_columns in schema.items():
        names = {c.name for c in ekorlap_columns}
        if not {"ekorlap", "tbszam"} <= names:
            continue
        ek_var, _ = columns_by_name(ekorlap_columns)
        for row in iter_rows(db, ekorlap_table):
            decoded = decode_row(row)
            if decoded is None:
                continue
            values = decoded[2]
            if max(ek_var.values()) >= len(values):
                continue
            note = rtf_to_text(read_memo(db, values[ek_var["ekorlap"]]))
            key = normalize_person_key(clean_text(values[ek_var["tbszam"]]), "")
            if note and key in pseudonyms:
                notes.append({"patient_id": pseudonyms[key], "note": note})
    return records, notes, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", required=True, help="path to PEDI3.MDB")
    parser.add_argument("--images", help="image folder; adds an image_file column for existing files")
    parser.add_argument("--out-dir", required=True, help="output folder for the CSV files")
    args = parser.parse_args()

    with open(args.db, "rb") as f:
        db = f.read()
    records, notes, skipped = extract_records(db)

    if args.images:
        files = {os.path.splitext(name)[0].upper(): name for name in os.listdir(args.images)}
        for record in records:
            record["image_file"] = files.get(record["image_id"], "")

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "pedis_records.csv"), records)
    # The notes are free text and may still mention people: keep this file local.
    write_csv(os.path.join(args.out_dir, "pedis_ekorlap_notes.csv"), notes)
    print(f"{len(records)} image records, {len(notes)} e-record notes written to {args.out_dir} "
          f"({skipped} malformed rows skipped)", file=sys.stderr)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
