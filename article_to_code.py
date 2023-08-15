import re
import openai
import PyPDF2
import requests
from io import BytesIO
from tqdm import tqdm


base_article_prompt = "Rewrite the presented approach with correct, academic grammar. Write as much of each proposed detail as possible and how to apply it. Technical and mathematical details are crucial. The generated text needs to be expository and informative. Do not remove important information. Remain unbiased, without critical or opinionated comments. Your text needs to expose the idea, explain it technically and every detail about how it works. The advantages of the approach don't matter, only how you apply it matters."
base_code_prompt = "Below is the text of an approach described in detail, as well as the code that needs that approach applied to it. This approach can be applied to code and accurately describes how this can be done. First, write how you will apply the approach in a nutshell. Then rewrite the code adding and changing as needed to include all the specifics of the approach. Not only, add parameters in the functions so that they are flexible. Don't describe how to apply the approach, just write the code. Ensures you are only using declared variables. The most important thing is to ensure that all the code is written, with no incomplete or missing parts."


class GPTTextGenerator:
    def __init__(self, gpt_model: str = "gpt-3.5-turbo-16k", temperature: float = 0, max_tokens: int = 256,
                 top_p: float = 0, frequency_penalty: float = 0, presence_penalty: float = 0):
        """
        Initializes the GPTTextGenerator class.

        Args:
            gpt_model (str): Name of the GPT model. Default: "text-davinci-003".
            temperature (float): Degree of randomness. Default: 0.
            max_tokens (int): Maximum response length in tokens. Default: 256.
            top_p (float): Diversity of response by capping cumulative probability. Default: 0.
            frequency_penalty (float): Controls repetition. Default: 0.
            presence_penalty (float): Controls relevance to input. Default: 0.
        """
        self.gpt_model = gpt_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def generate(self, prompt: str) -> str:
        """
        Generates a response to the prompt using GPT API.

        Args:
            prompt (str): Input prompt.

        Returns:
            str: Generated response or empty string on error.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a useful helper. You specialize in natural language processing, programming, and troubleshooting. You know how important the jobs you participate in are, and you dedicate yourself to being a perfectionist and attentive to detail."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error: {e}")
            return ""


def fetch_pdf_text(pdf_url: str) -> str:
    """
    Fetches the text content from a PDF URL.

    Args:
        pdf_url (str): The URL of the PDF.

    Returns:
        str: The extracted text from the PDF.
    """
    # Send a request to the PDF URL and retrieve the PDF content as bytes
    response = requests.get(pdf_url)
    pdf_bytes = BytesIO(response.content)

    # Extract text from the PDF using PyPDF2
    pdf_text = ''
    pdf_reader = PyPDF2.PdfReader(pdf_bytes)

    # Iterate through each page and extract text
    for page in tqdm(pdf_reader.pages):
        pdf_text += page.extract_text()

    return pdf_text


def extract_and_filter_text(text: str, start_marker: str = "abstract", end_marker: str = "references",
                            apply_filter: bool = False, min_chars: int = 2) -> str:
    """
    Extracts and optionally filters text between specified markers from the input text.

    Args:
        text (str): The input text.
        start_marker (str, optional): The marker indicating the beginning of the text to extract. Defaults to "abstract".
        end_marker (str, optional): The marker indicating the end of the text to extract. Defaults to "references".
        apply_filter (bool, optional): Whether to apply alphabetic line filtering. Defaults to False.
        min_chars (int, optional): The minimum number of alphabetic characters required per line for filtering.
                                  Defaults to 2.

    Returns:
        str: The text extracted between the specified markers, optionally filtered based on the conditions.
    """
    start_found = False
    extracted_text = ''
    lines = text.splitlines(True)

    for line in tqdm(lines):
        if not start_found:
            if start_marker.lower() in line.lower():
                start_found = True
        elif end_marker.lower() in line.lower():
            break
        elif not apply_filter or (apply_filter and sum(c.isalpha() for c in line) >= min_chars):
            extracted_text += line

    return extracted_text


def clean_non_alphabetic_items(text: str) -> str:
    """
    Removes non-alphabetic items (excluding digits) from the input text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with non-alphabetic items removed.
    """
    items = text.split()
    return ' '.join([item for item in tqdm(items) if any(c.isalpha() for c in item) or item.isdigit()])


def replace_urls_with_placeholder(text: str) -> str:
    """
    Replaces URLs in the input text with a placeholder.

    Args:
        text (str): The input text.

    Returns:
        str: The text with URLs replaced by a placeholder.
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, 'link', text)


def start_of_code(code_text: str, start_keyword: str = "import") -> str:
    """
    Returns the portion of code starting from a specified keyword.

    Args:
        code_text (str): The input code text.
        start_keyword (str, optional): The keyword to start from. Defaults to "import".

    Returns:
        str: The code text starting from the keyword.
    """
    code_lines = code_text.splitlines(True)
    start_found = False
    final_code = ""

    # Iterate through lines and extract code starting from the keyword
    for line in tqdm(code_lines):
        if line.startswith(start_keyword):
            start_found = True
        if start_found:
            final_code += line
    return final_code


def generate_updated_code(pdf_url: str, target_file: str, final_name: str,
                          start_marker: str = "abstract", end_marker: str = "references",
                          min_chars: int = 2, apply_filter: bool = True, start_keyword: str = "import",
                          gpt_model: str = "gpt-3.5-turbo-16k", temperature: float = 0,
                          max_tokens: int = 4096, top_p: float = 0, frequency_penalty: float = 0,
                          presence_penalty: float = 0) -> str:
    """
    Generates and writes updated code based on a PDF article and existing code.

    Args:
        pdf_url (str): URL of the PDF article.
        target_file (str): Path to the target code file.
        final_name (str): Path to the final updated code file.
        start_marker (str, optional): The starting marker for text extraction. Defaults to "abstract".
        end_marker (str, optional): The ending marker for text extraction. Defaults to "references".
        min_chars (int, optional): The minimum number of alphabetic characters in filtered lines. Defaults to 2.
        apply_filter (bool, optional): Whether to apply alphabetic line filtering. Defaults to True.
        start_keyword (str, optional): The keyword to start code extraction. Defaults to "import".
        gpt_model (str, optional): Name of the GPT model. Defaults to "gpt-3.5-turbo-16k".
        temperature (float, optional): Degree of randomness for GPT response. Defaults to 0.
        max_tokens (int, optional): Maximum response length in tokens. Defaults to 4096.
        top_p (float, optional): Diversity of response by capping cumulative probability. Defaults to 0.
        frequency_penalty (float, optional): Controls repetition in GPT response. Defaults to 0.
        presence_penalty (float, optional): Controls relevance to input in GPT response. Defaults to 0.

    Returns:
        str: The generated updated code.
    """
    # Fetch text content from the PDF URL
    pdf_text = fetch_pdf_text(pdf_url)

    # Extract the text between the specified headings
    extracted_text = extract_and_filter_text(pdf_text, start_marker=start_marker, end_marker=end_marker,
                                             apply_filter=apply_filter, min_chars=min_chars)

    # Remove non-alphabetic and non-alphabetic items from text
    cleaned_text = clean_non_alphabetic_items(extracted_text)

    # Replace URLs with a placeholder ("link") in the text
    article_text = replace_urls_with_placeholder(cleaned_text)

    # Create the prompt for the article generation
    article_prompt = f"{base_article_prompt}\n\n{article_text}"

    # Initialize the GPTTextGenerator with provided parameters
    model = GPTTextGenerator(gpt_model=gpt_model, temperature=temperature, max_tokens=max_tokens,
                             top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

    # Generate the rewritten approach text for the article
    approach_text = model.generate(article_prompt).replace("\n", " ")

    # Open and read the code of the target file
    with open(target_file, "r") as f:
        code_text = f.read()

    # Remove excessive newlines (consecutive newlines) from the code text
    code_text = re.sub(r'\n+', '\n', code_text)

    # Create the prompt for the code generation
    code_prompt = f"{base_code_prompt}\n\nThe code:\n\n{code_text}\n\nThe approach:\n\n{approach_text}"

    # Generate the updated code with the integrated approach
    updated_code = model.generate(code_prompt)

    # Get the relevant portion of the updated code
    final_code = start_of_code(updated_code, start_keyword=start_keyword)

    # Write the updated code to the final file
    with open(final_name, "w") as f:
        f.write(final_code)

    return final_code
