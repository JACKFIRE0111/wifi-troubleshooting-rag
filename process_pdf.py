from pypdf import PdfReader
import json
import re

PDF_FILE = "troubleshoot-windows-client-WiFi-Research-Simplified.pdf"
OUTPUT_FILE = "wifi_chunks.json"

reader = PdfReader(PDF_FILE)

# ============================================================
# 1. Extract PDF text
# ============================================================

full_text = ""

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    if text:
        full_text += f"\n[PAGE {page_number}]\n"
        full_text += text


# ============================================================
# 2. Exact 16 sections
# ============================================================

sections = [
    (1, "What was retained"),
    (2, "What was removed"),
    (3, "Core symptom taxonomy"),
    (4, "Baseline evidence collection"),
    (5, "Standard wireless commands / reports"),
    (6, "Windows Wi-Fi component model"),
    (7, "Troubleshooting logic for RAG"),
    (8, "802.1X / enterprise Wi-Fi authentication"),
    (9, "802.1X evidence collection"),
    (10, "Intermittent disconnects and roaming"),
    (11, "Wi-Fi associated but Internet/network access fails"),
    (12, "Selected Wi-Fi-specific legacy cases"),
    (13, "Captive-portal Wi-Fi"),
    (14, "Recommended RAG chunk structure"),
    (15, "Source map"),
    (16, "Research-use note")
]


# ============================================================
# 3. Find pages
# ============================================================

page_matches = []

for match in re.finditer(r"\[PAGE\s+(\d+)\]", full_text):

    page_matches.append({
        "page": int(match.group(1)),
        "position": match.start()
    })


# ============================================================
# 4. Find section headings
#
# Allow:
# 3. Core symptom taxonomy
# 3.
# Core symptom taxonomy
# 3.    Core symptom taxonomy
# ============================================================

section_matches = []

for number, title in sections:

    title_pattern = re.escape(title)

    # Spaces in title can also be newlines
    title_pattern = title_pattern.replace(
        r"\ ",
        r"\s+"
    )

    pattern = re.compile(
        rf"(?:^|\n)\s*{number}\.\s*{title_pattern}\s*(?=\n|$)",
        re.IGNORECASE
    )

    matches = list(pattern.finditer(full_text))

    if matches:

        match = matches[0]

        section_matches.append({
            "number": number,
            "title": title,
            "start": match.start()
        })

        print(f"[FOUND]   {number}. {title}")

    else:

        print(f"[MISSING] {number}. {title}")


# ============================================================
# 5. Sort strictly by section number
# ============================================================

section_matches.sort(
    key=lambda x: x["number"]
)


# ============================================================
# 6. Build chunks
# ============================================================

chunks = []

for i, section in enumerate(section_matches):

    start = section["start"]

    if i + 1 < len(section_matches):

        end = section_matches[i + 1]["start"]

    else:

        end = len(full_text)


    section_text = full_text[start:end].strip()


    # --------------------------------------------------------
    # Determine pages
    # --------------------------------------------------------

    pages_used = []

    for j, page in enumerate(page_matches):

        page_start = page["position"]

        if j + 1 < len(page_matches):

            next_page_start = page_matches[j + 1]["position"]

        else:

            next_page_start = len(full_text)


        if next_page_start > start and page_start < end:

            pages_used.append(page["page"])


    pages_used = sorted(set(pages_used))


    # --------------------------------------------------------
    # Remove page markers
    # --------------------------------------------------------

    clean_text = re.sub(
        r"\[PAGE\s+\d+\]",
        "",
        section_text
    )

    clean_text = re.sub(
        r"\s+",
        " ",
        clean_text
    ).strip()


    # --------------------------------------------------------
    # Keep section
    # --------------------------------------------------------

    if len(clean_text) < 50:

        print(
            f"[WARNING] Section {section['number']} "
            f"is unusually short: {len(clean_text)} chars"
        )

        


    chunks.append({

        "chunk_id": f"section_{section['number']:03d}",

        "section": section["title"],

        "pages": pages_used,

        "text": clean_text
    })


# ============================================================
# 7. Save
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 8. Final verification
# ============================================================

print("\n" + "=" * 70)

print(f"Pages processed: {len(reader.pages)}")

print(f"Section chunks created: {len(chunks)}")

print(f"Saved to: {OUTPUT_FILE}")

print("=" * 70)

print("\nSections:")

for chunk in chunks:

    print(
        f"{chunk['chunk_id']} | "
        f"{chunk['section']} | "
        f"pages={chunk['pages']}"
    )


# ============================================================
# 9. Check whether all 16 sections exist
# ============================================================

expected_numbers = set(range(1, 17))

actual_numbers = set(
    int(chunk["chunk_id"].split("_")[1])
    for chunk in chunks
)

missing_numbers = sorted(
    expected_numbers - actual_numbers
)

print("\nVerification:")

if not missing_numbers:

    print("SUCCESS: All 16 sections were created.")

else:

    print(
        "MISSING SECTION NUMBERS:",
        missing_numbers
    )