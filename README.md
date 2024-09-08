# **LangChain Multi-Source Q&A Project**

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** question-answering system using multiple sources like **Wikipedia**, **Arxiv**, and **LangSmith** documentation. It leverages **LangChain** and **OpenAI GPT** to provide context-aware responses to user queries.

## **Features**
- Query **Wikipedia** and **Arxiv** for real-time information retrieval.
- Use **LangSmith search** for LangChain-specific documentation or you may use any webpage that want to extract the info from . 
- Integrated with **OpenAI's GPT** for enhanced natural language processing and question-answering.

## **Disclaimer**
Please note that using **OpenAI GPT** will incur **minimal charges** based on the number of API requests made. Ensure that you monitor your OpenAI usage to avoid unexpected costs.

## **Table of Contents**
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## **Installation**

### **Prerequisites**
- Python 3.9+
- OpenAI API key

### **Steps**
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/yourprojectname.git](https://github.com/amrokamalelsiddig/Multi-Source-Knowledge-Retrieval-System.git)
   cd Multi-Source-Knowledge-Retrieval-System


2. Set up environment variables:
   
    Rename .env.example to .env.
    Update the .env file with your OpenAI API key.

3. Install the required dependencies:
   ```bash
    pip install -r requirements.txt


### **Usage**

Once the project is set up, you can run it either through the Python script or the Jupyter notebook.
Running the Python Script

```bash
python app.py
```

Running the Jupyter Notebook

jupyter notebook project_notebook.ipynb

After running, you can start querying the system with questions related to LangSmith, Wikipedia, or Arxiv topics. Example query: "Tell me about LangSmith."
