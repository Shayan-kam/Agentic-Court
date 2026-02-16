import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
from fpdf import FPDF
import time

# --- 1. DATA EXTRACTION ---
print("Fetching data from NBA API...")

# Find LeBron James' ID
nba_players = players.find_players_by_full_name('LeBron James')
lebron_id = nba_players[0]['id']

# Fetch Career Stats
# We add a small timeout/retry logic because the NBA API can be moody
career = playercareerstats.PlayerCareerStats(player_id=lebron_id)
df = career.get_data_frames()[0]  # Index 0 is SeasonTotalsRegularSeason

# Select only the most important columns to ensure they fit on a PDF page
columns_to_keep = [
    'SEASON_ID', 'TEAM_ABBREVIATION', 'PLAYER_AGE', 'GP', 
    'GS', 'MIN', 'FGM', 'FG3M', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'PTS'
]
df_filtered = df[columns_to_keep]

print(f"Success! Retrieved {len(df_filtered)} seasons of data.")

# --- 2. PDF GENERATION ---
print("Generating PDF...")

class NBA_PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'LeBron James - Full Career Statistics (Regular Season)', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# Initialize PDF in Landscape ('L') to fit all columns
pdf = NBA_PDF(orientation='L', unit='mm', format='A4')
pdf.add_page()
pdf.set_font("Arial", size=10)

# Define column widths (Total width should be around 270mm for Landscape A4)
col_width = 19 

# Add Table Headers
pdf.set_fill_color(200, 200, 200) # Light grey background for header
pdf.set_font("Arial", 'B', 9)
for col in columns_to_keep:
    pdf.cell(col_width, 10, col, 1, 0, 'C', True)
pdf.ln()

# Add Table Data
pdf.set_font("Arial", size=9)
for i in range(len(df_filtered)):
    for col in columns_to_keep:
        # Get the value and convert to string
        val = str(df_filtered.iloc[i][col])
        pdf.cell(col_width, 10, val, 1, 0, 'C')
    pdf.ln()

# Save the file
output_filename = "LeBron_James_Career_Stats.pdf"
pdf.output(output_filename)

print(f"Done! Your file '{output_filename}' is ready.")