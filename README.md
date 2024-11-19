# Priority Text  
**Your All-in-One Text Intelligence API**  

---  

## Overview  

**Priority Text** is a versatile API that combines state-of-the-art spam detection with intelligent text summarization capabilities. It ensures clean, efficient communication by identifying unwanted messages and generating concise summaries, making it an indispensable tool for modern applications.  
**Access it here**: [Priority Text API Webpage](https://prioritytext-api.onrender.com/)  

---  

## Key Features  

- **Dual-Purpose Functionality**: Offers precise spam detection with 99% precision and 98% accuracy, along with text summarization powered by Hugging Face T5-small.  
- **Machine Learning Excellence**: Utilizes a robust Random Forest classifier for spam detection and advanced NLP techniques for summarization.  
- **Developer-Friendly API**: Simple integration with fast response times (under 2 seconds) for seamless text processing.  

---  

## Development Process  

1. **Data Cleaning**: Curated a 5,572-row dataset of spam and ham messages for training, ensuring diverse linguistic representation.  
2. **Preprocessing**: Applied NLP techniques like stop word removal, stemming, and tokenization to extract meaningful patterns.  
3. **Feature Engineering**: Leveraged TF-IDF vectorization and additional features, boosting spam detection accuracy by 2%.  
4. **Text Summarization**: Integrated Hugging Face T5-small to generate concise summaries for submitted text.  
5. **Deployment**: Deployed on Render, providing public access with efficient backend handling via Flask.  

---  

## Usage  

- **Input**:  
  ```json  
  {  
      "text": "Your message here",  
      "min_length": 10,    // Optional  
      "max_length": 150    // Optional  
  }  

## Output  

```json  
{  
    "result": "Ham",  
    "summary": "Your text summary",  
    "confidence": 0.98  
}  