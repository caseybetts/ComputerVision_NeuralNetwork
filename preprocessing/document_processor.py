import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

class DocumentProcessor:
    def __init__(self, target_size=(28, 28)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
    
    def load_document(self, image_path):
        """Load and preprocess a document image"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path
            
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        return image
    
    def preprocess_document(self, image):
        """Preprocess document for better text extraction"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Remove noise
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Invert (white text on black background)
        binary = cv2.bitwise_not(binary)
        
        return binary
    
    def detect_text_regions(self, image, min_area=100):
        """Detect regions containing text"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio to avoid very wide or tall regions
                aspect_ratio = w / h
                if 0.1 < aspect_ratio < 10:
                    text_regions.append((x, y, w, h))
        
        # Sort regions by position (top to bottom, left to right)
        text_regions.sort(key=lambda x: (x[1], x[0]))
        
        return text_regions
    
    def segment_characters(self, text_region, image):
        """Segment individual characters from a text region"""
        x, y, w, h = text_region
        region = image[y:y+h, x:x+w]
        
        # Find character contours
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characters = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 20:  # Minimum character area
                cx, cy, cw, ch = cv2.boundingRect(contour)
                char_region = region[cy:cy+ch, cx:cx+cw]
                
                # Add padding and resize to target size
                char_processed = self.prepare_character(char_region)
                if char_processed is not None:
                    characters.append({
                        'image': char_processed,
                        'bbox': (cx + x, cy + y, cw, ch),
                        'area': area
                    })
        
        # Sort characters by x-position
        characters.sort(key=lambda x: x['bbox'][0])
        
        return characters
    
    def prepare_character(self, char_image):
        """Prepare a character image for the neural network"""
        try:
            # Add padding to make it square
            h, w = char_image.shape
            size = max(h, w)
            padded = np.zeros((size, size), dtype=np.uint8)
            
            # Center the character
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            padded[y_offset:y_offset+h, x_offset:x_offset+w] = char_image
            
            # Resize to target size
            resized = cv2.resize(padded, self.target_size)
            
            # Convert to PIL and apply transforms
            pil_image = Image.fromarray(resized)
            tensor = self.transform(pil_image)
            
            return tensor
            
        except Exception as e:
            print(f"Error preparing character: {e}")
            return None
    
    def visualize_document_processing(self, image, text_regions, characters=None):
        """Visualize the document processing steps"""
        # Create a copy for visualization
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw text regions
        for i, (x, y, w, h) in enumerate(text_regions):
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_image, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw character bounding boxes if provided
        if characters:
            for char in characters:
                x, y, w, h = char['bbox']
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(vis_image)
        plt.title('Document Processing Visualization')
        plt.axis('off')
        plt.show()
