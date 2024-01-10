from dotenv import load_dotenv
from src import paper_to_code as ptc

# Load environment variables from the .env file
load_dotenv()

# URL to Cyclical Learning Rates paper
pdf_url = ""

# MNIST base code
target_file = ""

# Generated python file name
final_name = ""

# Call the function to generate and save the updated code
ptc.generate_updated_code(pdf_url, target_file, final_name)
