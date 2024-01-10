from dotenv import load_dotenv
from src import paper_to_code as ptc

# Load environment variables from the .env file
load_dotenv()

# URL to Layer Normalization paper
pdf_url = 'https://arxiv.org/pdf/1607.06450.pdf'

# MNIST base code
target_file = "mnist_code.py"

# Generated python file name
final_name = "updated_mnist_code.py"

# Call the function to generate and save the updated code
ptc.generate_updated_code(pdf_url, target_file, final_name, start_marker="2 Background", end_marker="6 Experimental results")