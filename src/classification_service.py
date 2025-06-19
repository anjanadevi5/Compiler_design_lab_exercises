import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pdf2image import convert_from_path # For PDF processing
import os
import argparse

# --- Configuration ---
MODEL_PATH = 'resnet50_passport_dl_classifier.pth' # Ensure this is in the same directory or provide full path
CLASS_NAMES = ['dl', 'passport'] # IMPORTANT: Must match training order
NUM_CLASSES = len(CLASS_NAMES)

# --- Image Transformations ---
# IMPORTANT: These MUST match the transformations used during your training.
def get_image_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Model Loading ---
def load_trained_model(model_path, num_classes):
    """Loads the ResNet50 model structure and your trained weights."""
    model = models.resnet50(weights=None) # Start with an uninitialized model

    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features
    # Replace the last fully connected layer to match your number of classes
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the state dictionary
    # Ensure the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. "
                                f"Ensure it's in the correct location or update MODEL_PATH.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode
    return model, device

# --- Preprocessing Functions ---
def preprocess_pil_image(pil_image, transform):
    """Preprocesses a PIL Image object and returns the tensor."""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    image_tensor = transform(pil_image).unsqueeze(0) # Add batch dimension
    return image_tensor

def preprocess_image_from_path(image_path, transform):
    """Loads an image from path, preprocesses it, and returns the tensor."""
    try:
        image = Image.open(image_path)
        return preprocess_pil_image(image, transform)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening or processing image {image_path}: {e}")
        return None

def preprocess_pdf_first_page_from_path(pdf_path, transform):
    """Extracts the first page of a PDF, preprocesses it, and returns the tensor."""
    try:
        # You might need to specify poppler_path if pdf2image can't find it
        # e.g., images = convert_from_path(pdf_path, first_page=1, last_page=1, poppler_path='/usr/bin/poppler')
        images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200) # dpi can be adjusted
        if not images:
            print(f"Error: Could not extract any pages from PDF: {pdf_path}")
            return None
        first_page_pil_image = images[0]
        return preprocess_pil_image(first_page_pil_image, transform)
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e: # Catching pdf2image specific errors might be more robust
        print(f"Error processing PDF {pdf_path}: {e}")
        print("Ensure 'poppler' is installed and in your PATH, or specify 'poppler_path' in convert_from_path.")
        return None

# --- Prediction Function ---
def predict(model, device, image_tensor, class_names):
    """Uses the model to predict the class of the preprocessed image tensor."""
    if image_tensor is None:
        return None, None

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class_name = class_names[predicted_idx.item()]
    return predicted_class_name, confidence.item()

# --- Main Service Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification Service: Preprocess and predict class for an image or PDF.")
    parser.add_argument("file_path", type=str, help="Path to the image or PDF file to classify.")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: Input file not found at '{args.file_path}'")
        exit()

    print("Loading model...")
    try:
        model, device = load_trained_model(MODEL_PATH, NUM_CLASSES)
        print(f"Model loaded successfully. Using device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    image_transformations = get_image_transforms()
    processed_tensor = None
    file_extension = os.path.splitext(args.file_path)[1].lower()

    print(f"\nPreprocessing '{args.file_path}'...")
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']:
        processed_tensor = preprocess_image_from_path(args.file_path, image_transformations)
    elif file_extension == '.pdf':
        processed_tensor = preprocess_pdf_first_page_from_path(args.file_path, image_transformations)
    else:
        print(f"Error: Unsupported file type '{file_extension}'. Please provide a supported image or a PDF.")
        exit()

    if processed_tensor is not None:
        print("Preprocessing complete. Predicting...")
        predicted_class, confidence_score = predict(model, device, processed_tensor, CLASS_NAMES)
        if predicted_class:
            print(f"\n--- Prediction Result ---")
            print(f"File: {os.path.basename(args.file_path)}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence_score:.4f}")
        else:
            print("Prediction failed.")
    else:
        print("Failed to preprocess the input file.")
