import argparse
import csv
import os
import sys
import joblib
import numpy as np
import re
import nltk
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class SemanticSimilarityCLI:
    def __init__(self, model_path="models"):
        self.model_path = model_path
        self.encoder = None
        self.classifier = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.load_models()
        
    def load_models(self):
        """Load pretrained models from disk"""
        try:
            logger.info("Loading models...")
            
            # Verify paths
            encoder_path = os.path.join(self.model_path, "semantic_encoder")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder folder not found: {encoder_path}")
                
            model_file = os.path.join(self.model_path, "semantic_similarity_model.pkl")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            # Load models
            self.encoder = SentenceTransformer(encoder_path)
            self.classifier = joblib.load(model_file)
            
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {str(e)}")
            logger.error("Please ensure:")
            logger.error("1. You have the 'models' folder")
            logger.error("2. It contains 'semantic_encoder/' folder and 'semantic_similarity_model.pkl' file")
            sys.exit(1)
            
    def preprocess(self, text):
        """Clean and normalize text"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = text.lower().strip()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters
        text = re.sub(r'[^\w\s.,?!;:\'"-]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def predict_similarity(self, sentence1, sentence2):
        """Predict similarity between two sentences"""
        try:
            # Preprocess
            s1_clean = self.preprocess(sentence1)
            s2_clean = self.preprocess(sentence2)
            
            # Skip if empty
            if not s1_clean or not s2_clean:
                return "No", 0.0
                
            # Generate embeddings
            vec1 = self.encoder.encode([s1_clean], convert_to_tensor=True)
            vec2 = self.encoder.encode([s2_clean], convert_to_tensor=True)
            
            # Calculate cosine similarity
            cosine_score = util.cos_sim(vec1, vec2).item()
            
            # Scale to [0, 1] for probability
            prob_score = (cosine_score + 1) / 2
            
            # Predict using classifier
            feature = np.array([[prob_score]])
            prediction = self.classifier.predict(feature)[0]
            confidence = self.classifier.predict_proba(feature)[0][1] * 100
                
            return ("Yes" if prediction == 1 else "No", round(confidence, 2))
        except Exception as e:
            logger.error(f"⚠️ Prediction error: {str(e)}")
            return "Error", 0.0

    def process_batch(self, input_file, output_file):
        """Process a CSV file containing sentence pairs"""
        if not os.path.exists(input_file):
            logger.error(f"❌ Input file not found: {input_file}")
            return False
            
        try:
            with open(input_file, 'r') as f:
                reader = csv.DictReader(f)
                if 'sentence1' not in reader.fieldnames or 'sentence2' not in reader.fieldnames:
                    logger.error("❌ CSV must contain 'sentence1' and 'sentence2' columns")
                    return False
                    
                rows = list(reader)
                
            logger.info(f"🔍 Processing {len(rows)} sentence pairs...")
            results = []
            for i, row in enumerate(rows):
                result, confidence = self.predict_similarity(
                    row['sentence1'], row['sentence2']
                )
                row['similar'] = result
                row['confidence'] = f"{confidence}%"
                results.append(row)
                
                # Print progress every 10% or 100 items
                if (i + 1) % max(1, len(rows)//10) == 0 or (i + 1) == len(rows):
                    logger.info(f"🔄 Processed {i+1}/{len(rows)} pairs")
            
            # Write results
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
                
            logger.info(f"💾 Results saved to {output_file}")
            logger.info(f"🎉 Successfully processed {len(results)} pairs")
            return True
            
        except Exception as e:
            logger.error(f"🔥 Batch processing error: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description='Semantic Similarity Detection CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s1', '--sentence1', help='First sentence for comparison')
    parser.add_argument('-s2', '--sentence2', help='Second sentence for comparison')
    parser.add_argument('-f', '--file', help='CSV file for batch processing')
    parser.add_argument('-o', '--output', help='Output file for batch results')
    parser.add_argument('-m', '--model-path', default="models", 
                        help='Path to model directory')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Create detector instance
    detector = SemanticSimilarityCLI(args.model_path)
    
    if args.file:
        output = args.output or "similarity_results.csv"
        success = detector.process_batch(args.file, output)
        sys.exit(0 if success else 1)
    elif args.sentence1 and args.sentence2:
        result, confidence = detector.predict_similarity(args.sentence1, args.sentence2)
        print(f"\n🔤 Sentence 1: {args.sentence1}")
        print(f"🔤 Sentence 2: {args.sentence2}")
        print(f"🤝 Similar: {result}")
        print(f"📊 Confidence: {confidence}%")
    else:
        logger.error("❌ Error: Please provide either two sentences or an input CSV file")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()