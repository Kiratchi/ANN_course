from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Input and output files
inp = "reservoir.py"
out = "reservoir.pdf"

# Create a PDF canvas
c = canvas.Canvas(out, pagesize=letter)
width, height = letter

# Read the Python file
with open(inp, "r") as f:
    code_lines = f.readlines()

# Set font
c.setFont("Courier", 10)
y = height - 50  # start position from top

# Write each line to the PDF
for line in code_lines:
    if y < 50:  # start a new page if space runs out
        c.showPage()
        c.setFont("Courier", 10)
        y = height - 50
    c.drawString(50, y, line.rstrip())
    y -= 12  # move down line by line

c.save()
print(f"Saved {out}")