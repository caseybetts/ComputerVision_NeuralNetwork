import torch
import torch.nn.functional as F
import numpy as np
from preprocessing.document_processor import DocumentProcessor
import re

class DocumentReader:
    def __init__(self, model_path, device='cuda', dataset_type='letters'):
        self.device = device
        self.processor = DocumentProcessor()
        self.dataset_type = dataset_type
        
        # Load the trained model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Character mapping based on dataset type
        if dataset_type == 'letters':
            self.char_mapping = {i: chr(65 + i) for i in range(26)}  # A-Z
        elif dataset_type == 'digits':
            self.char_mapping = {i: str(i) for i in range(10)}  # 0-9
        elif dataset_type == 'balanced':
            # 0-9, then A-Z
            self.char_mapping = {i: str(i) for i in range(10)}
            self.char_mapping.update({i: chr(55 + i) for i in range(10, 36)})  # A-Z
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Post-processing rules for letters
        self.post_process_rules = [
            # Common OCR corrections for letters
            ('0', 'O'),  # Often confused
            ('1', 'l'),  # Often confused
            ('5', 'S'),  # Often confused
            ('8', 'B'),  # Often confused
            ('I', 'l'),  # Often confused
            ('O', '0'),  # Often confused
            ('S', '5'),  # Often confused
            ('B', '8'),  # Often confused
        ]
    
    def load_model(self, model_path):
        """Load the trained model"""
        from models.cnn_models import EMNISTCNN
        
        # Determine number of classes based on dataset type
        if self.dataset_type == 'letters':
            num_classes = 26
        elif self.dataset_type == 'digits':
            num_classes = 10
        elif self.dataset_type == 'balanced':
            num_classes = 36
        else:
            num_classes = 26  # Default to letters
        
        model = EMNISTCNN(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model
    
    def predict_character(self, char_tensor):
        """Predict a single character"""
        with torch.no_grad():
            char_tensor = char_tensor.unsqueeze(0).to(self.device)
            output = self.model(char_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1)
            confidence = probabilities[0][prediction].item()
            
        return prediction.item(), confidence
    
    def read_document(self, image_path, visualize=False):
        """Read an entire document and return the text"""
        # Load and preprocess document
        image = self.processor.load_document(image_path)
        processed_image = self.processor.preprocess_document(image)
        
        # Detect text regions
        text_regions = self.processor.detect_text_regions(processed_image)
        
        if visualize:
            self.processor.visualize_document_processing(processed_image, text_regions)
        
        # Process each text region
        document_text = []
        for region_idx, region in enumerate(text_regions):
            # Segment characters
            characters = self.processor.segment_characters(region, processed_image)
            
            # Recognize characters
            line_text = []
            for char in characters:
                prediction, confidence = self.predict_character(char['image'])
                
                # Apply confidence threshold
                if confidence > 0.5:  # Adjust threshold as needed
                    character = self.char_mapping.get(prediction, '?')
                    line_text.append(character)
                else:
                    line_text.append('?')  # Low confidence character
            
            # Join characters into line
            line = ''.join(line_text)
            document_text.append(line)
        
        # Post-process the text
        processed_text = self.post_process_text(document_text)
        
        return {
            'text': processed_text,
            'raw_lines': document_text,
            'num_regions': len(text_regions),
            'num_characters': sum(len(line) for line in document_text)
        }
    
    def post_process_text(self, lines):
        """Apply post-processing rules to improve text quality"""
        processed_lines = []
        
        for line in lines:
            # Apply character corrections
            for old_char, new_char in self.post_process_rules:
                line = line.replace(old_char, new_char)
            
            # Remove excessive spaces
            line = re.sub(r'\s+', ' ', line).strip()
            
            # Remove lines that are too short (likely noise)
            if len(line) > 1:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def read_document_with_confidence(self, image_path, min_confidence=0.7):
        """Read document with confidence scores for each character"""
        # Load and preprocess document
        image = self.processor.load_document(image_path)
        processed_image = self.processor.preprocess_document(image)
        
        # Detect text regions
        text_regions = self.processor.detect_text_regions(processed_image)
        
        # Process each text region
        document_results = []
        for region_idx, region in enumerate(text_regions):
            # Segment characters
            characters = self.processor.segment_characters(region, processed_image)
            
            # Recognize characters with confidence
            line_results = []
            for char in characters:
                prediction, confidence = self.predict_character(char['image'])
                
                if confidence >= min_confidence:
                    character = self.char_mapping.get(prediction, '?')
                    line_results.append({
                        'character': character,
                        'confidence': confidence,
                        'bbox': char['bbox']
                    })
                else:
                    line_results.append({
                        'character': '?',
                        'confidence': confidence,
                        'bbox': char['bbox']
                    })
            
            document_results.append({
                'line_index': region_idx,
                'characters': line_results,
                'text': ''.join([c['character'] for c in line_results]),
                'avg_confidence': np.mean([c['confidence'] for c in line_results])
            })
        
        return document_results
