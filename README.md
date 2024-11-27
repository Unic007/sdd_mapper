# SDD Mapping

SDD Mapping is a Streamlit application designed to generate and visualize text embeddings using various NLP techniques.

## Quick Run
```sh
chmod +x run_sdd_mapping.sh
sh ./run_sdd_mapping.sh
```

## Setup

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/unicoconnect/sdd_mapping.git
    cd medmap
    ```

2. **Create a virtual environment:**

    ```sh
    python3 -m venv venv
    ```

3. **Activate the virtual environment:**

    On macOS and Linux:
    ```sh
    source venv/bin/activate
    ```

    On Windows:
    ```sh
    venv\Scripts\activate
    ```

4. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

5. **Install SpaCy and download the English language model:**

    ```sh
    pip install spacy
    python -m spacy download en_core_web_sm
    ```

6. **Accept the Xcode license (macOS only):**

    ```sh
    sudo xcodebuild -license
    ```

## Running the Application

To run the Streamlit application, execute the following command:

```sh
streamlit run app.py