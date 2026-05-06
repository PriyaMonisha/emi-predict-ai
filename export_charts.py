import os
from PIL import Image

PROJECT_ROOT = os.getcwd()
REPORTS_PATH = os.path.join(PROJECT_ROOT, 'data', 'reports')
OUTPUT_PDF   = os.path.join(PROJECT_ROOT, 'data', 'reports', 'EDA_Report.pdf')

# Get all EDA charts in order
chart_files = sorted([
    os.path.join(REPORTS_PATH, f)
    for f in os.listdir(REPORTS_PATH)
    if f.startswith('eda_') and f.endswith('.png')
])

print(f"Found {len(chart_files)} charts:")
for f in chart_files:
    print(f"  {os.path.basename(f)}")

# Convert to PDF
images = []
for chart in chart_files:
    img = Image.open(chart).convert('RGB')
    images.append(img)

if images:
    images[0].save(
        OUTPUT_PDF,
        save_all=True,
        append_images=images[1:]
    )
    size_mb = os.path.getsize(OUTPUT_PDF) / (1024*1024)
    print(f"\n✅ PDF created: {OUTPUT_PDF}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Pages: {len(images)}")

    # Open PDF automatically
    os.startfile(OUTPUT_PDF)
else:
    print("❌ No charts found!")