# =========================================================
# FACE RECOGNITION SYSTEM WITH PCA & TEAM MEMBER ASSIGNMENTS
# =========================================================

import os
import cv2
import numpy as np
import face_recognition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import time


# ============================================================
# TEAM MEMBER 1: AMIN - DATASET MANAGEMENT & ORGANIZATION
# ============================================================
# Responsibilities:
# - Creating and managing the dataset structure
# - Preprocessing face images
# - Extracting face encodings from images
# - Documentation of dataset requirements
# ============================================================
class DataProcessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.face_encodings = []
        self.face_labels = []
        self.person_names = []
        self.label_to_name = {}

    def load_dataset(self):
        """Load and preprocess images from the dataset folder"""
        print("Loading face dataset...")

        # AMIN: Verify dataset directory exists
        if not os.path.exists(self.dataset_path):
            print(f"Error: Dataset path '{self.dataset_path}' not found.")
            return False

        # AMIN: Get all person folders in the dataset
        person_folders = [f for f in os.listdir(self.dataset_path)
                          if os.path.isdir(os.path.join(self.dataset_path, f))]

        if not person_folders:
            print("No person folders found in the dataset.")
            return False

        # AMIN: Assign numeric labels to each person
        for idx, person_name in enumerate(person_folders):
            self.person_names.append(person_name)
            self.label_to_name[idx] = person_name

        # AMIN: Process each person's folder
        for label, person_name in enumerate(person_folders):
            person_dir = os.path.join(self.dataset_path, person_name)
            image_files = [f for f in os.listdir(person_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                print(f"No images found for {person_name}")
                continue

            print(f"Processing {len(image_files)} images for {person_name}")

            # AMIN: Process each image file for this person
            for img_file in image_files:
                img_path = os.path.join(person_dir, img_file)
                try:
                    # AMIN: Load image and extract face encoding
                    image = face_recognition.load_image_file(img_path)
                    face_locations = face_recognition.face_locations(image)

                    if face_locations:
                        face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                        self.face_encodings.append(face_encoding)
                        self.face_labels.append(label)
                    else:
                        print(f"No face found in {img_path}")

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        print(f"Dataset loaded successfully. Processed {len(self.face_encodings)} face images.")
        return len(self.face_encodings) > 0

    def get_preprocessed_data(self):
        """Return the preprocessed data"""
        return np.array(self.face_encodings), np.array(self.face_labels), self.label_to_name


# ============================================================
# TEAM MEMBER 2: AYTAI - PCA IMPLEMENTATION & ANALYSIS
# ============================================================
# Responsibilities:
# - Implementing PCA dimensionality reduction
# - Standardizing face encodings
# - Visualizing eigenfaces and explained variance
# - Documentation of PCA mathematics and theory
# ============================================================
class PCAModel:
    def __init__(self, n_components=50):
        # AYTAI: Setting up the PCA model with components
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        """Fit PCA model and transform the data"""
        # AYTAI: Standardize the data before PCA
        X_scaled = self.scaler.fit_transform(X)

        # AYTAI: Apply PCA dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)

        # AYTAI: Print explained variance information
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        print(f"PCA with {self.n_components} components:")
        print(f"Total variance explained: {sum(explained_variance) * 100:.2f}%")
        print(f"First component explains {explained_variance[0] * 100:.2f}% of variance")
        print(f"First 5 components explain {cumulative_variance[4] * 100:.2f}% of variance")

        return X_pca

    def transform(self, X):
        """Transform new data using the fitted PCA model"""
        # AYTAI: Transform new face encodings with existing PCA model
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def inverse_transform(self, X_pca):
        """Inverse transform to reconstruct original features from PCA space"""
        # AYTAI: Convert PCA representation back to original space
        X_scaled = self.pca.inverse_transform(X_pca)
        return self.scaler.inverse_transform(X_scaled)

    def get_principal_components(self):
        """Get the principal components"""
        return self.pca.components_

    def get_eigenfaces(self, shape=(64, 64)):
        """Convert principal components to eigenfaces for visualization"""
        # AYTAI: Generate eigenfaces from principal components
        components = self.pca.components_
        eigenfaces = []

        for component in components[:10]:  # Get first 10 eigenfaces
            eigenface = self.scaler.inverse_transform([component])[0]
            # Normalize to 0-255 range for visualization
            eigenface = 255 * (eigenface - np.min(eigenface)) / (np.max(eigenface) - np.min(eigenface))
            eigenfaces.append(eigenface.astype(np.uint8))

        return eigenfaces


# ============================================================
# TEAM MEMBER 3: ANAS - CLASSIFIER IMPLEMENTATION
# ============================================================
# Responsibilities:
# - Implementing face recognition classifier
# - Testing and evaluating classifier performance
# - Fine-tuning recognition threshold
# - Documentation of recognition approach
# ============================================================
class FaceClassifier:
    def __init__(self):
        # ANAS: Set initial recognition threshold
        self.threshold = 0.6  # Threshold for face recognition

    def train(self, X_train, y_train):
        """Train the classifier - in this case, store the features and labels"""
        # ANAS: Store training data for nearest-neighbor classification
        self.X_train = X_train
        self.y_train = y_train
        print(f"Classifier trained with {len(X_train)} samples")

    def predict(self, X):
        """Predict the class of a face encoding"""
        # ANAS: Handle empty training data case
        if len(self.X_train) == 0:
            return -1

        # ANAS: Calculate face distances for nearest-neighbor recognition
        face_distances = face_recognition.face_distance(self.X_train, X)

        # ANAS: Find the closest match (minimum distance)
        best_match_index = np.argmin(face_distances)

        # ANAS: Check if match is close enough using threshold
        if face_distances[best_match_index] < self.threshold:
            return self.y_train[best_match_index]
        else:
            return -1  # Unknown face

    def evaluate(self, X_test, y_test):
        """Evaluate the classifier on test data"""
        # ANAS: Test classifier performance
        predictions = []
        for face_encoding in X_test:
            predictions.append(self.predict(face_encoding))

        # ANAS: Calculate and display performance metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(report)


# ============================================================
# TEAM MEMBER 4: HAKIM - MODEL STORAGE & RETRIEVAL
# ============================================================
# Responsibilities:
# - Implementing model serialization and deserialization
# - Managing model file structure
# - Testing model loading/saving
# - Documentation of model persistence approach
# ============================================================

# HAKIM: Real-Time Recognition with Model Loading from Disk
class RealTimeRecognition:
    def __init__(self, pca_model, classifier, label_to_name):
        # HAKIM: Initialize with loaded models
        self.pca_model = pca_model
        self.classifier = classifier
        self.label_to_name = label_to_name

    def run(self):
        """Run real-time face recognition using webcam"""
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting real-time face recognition. Press 'q' to quit.")

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break

            # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and face encodings in the current frame
            start_time = time.time()
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Loop through each face found in the frame
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # HAKIM: Transform face encoding using loaded PCA model
                face_encoding_pca = self.pca_model.transform([face_encoding])[0]

                # HAKIM: Reconstruct face encoding from PCA space
                reconstructed_encoding = self.pca_model.inverse_transform([face_encoding_pca])[0]

                # HAKIM: Predict identity using the loaded classifier
                label = self.classifier.predict(reconstructed_encoding)

                if label >= 0:
                    name = self.label_to_name[label]
                else:
                    name = "Unknown"

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw a label with the name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

            # Calculate and display FPS
            end_time = time.time()
            fps = 1.0 / max(end_time - start_time, 0.001)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the resulting image
            cv2.imshow('Real-time Face Recognition with PCA', frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release webcam and close windows
        video_capture.release()
        cv2.destroyAllWindows()


# ============================================================
# TEAM MEMBER 5: ALTAIR - REAL-TIME RECOGNITION & UI
# ============================================================
# Responsibilities:
# - Implementing webcam integration
# - Creating UI elements for face recognition
# - Implementing real-time processing pipeline
# - Documentation of real-time performance
# ============================================================

# ALTAIR: Visualization module for model analysis and understanding
class Visualizer:
    def __init__(self, pca_model, label_to_name):
        self.pca_model = pca_model
        self.label_to_name = label_to_name

    def visualize_eigenfaces(self):
        """Visualize the eigenfaces (principal components)"""
        # ALTAIR: Visualize eigenfaces for better understanding
        eigenfaces = self.pca_model.get_eigenfaces()

        if not eigenfaces:
            print("No eigenfaces available to visualize.")
            return

        # Create a figure to display eigenfaces
        rows = 2
        cols = 5
        fig_size = (15, 6)

        fig = plt.figure(figsize=fig_size)
        for i in range(min(len(eigenfaces), rows * cols)):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(eigenfaces[i].reshape(64, 64), cmap='gray')
            ax.set_title(f"Eigenface {i + 1}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_explained_variance(self):
        """Plot the explained variance ratio"""
        # ALTAIR: Create plots to analyze PCA performance
        explained_variance = self.pca_model.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_variance) + 1), explained_variance)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Component')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        plt.legend()

        plt.tight_layout()
        plt.show()


# ============================================================
# MAIN APPLICATION MODULE - INTEGRATES ALL COMPONENTS
# ============================================================
def main():
    print("Face Recognition System with PCA")
    print("===============================")

    # Get dataset path from user
    dataset_path = input("Enter the path to your dataset folder: ")

    # Initialize modules
    try:
        # AMIN's component: Data Processing
        data_processor = DataProcessor(dataset_path)
        if not data_processor.load_dataset():
            print("Failed to load dataset. Exiting...")
            return

        face_encodings, face_labels, label_to_name = data_processor.get_preprocessed_data()

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            face_encodings, face_labels, test_size=0.2, random_state=42)

        # AYTAI's component: PCA Model
        n_components = min(50, len(X_train) - 1)  # Ensure we don't exceed the number of samples
        pca_model = PCAModel(n_components=n_components)

        # Transform training data with PCA
        X_train_pca = pca_model.fit_transform(X_train)

        # ANAS's component: Classifier
        classifier = FaceClassifier()

        # Train classifier on original space data (for face_recognition compatibility)
        classifier.train(X_train, y_train)

        # Evaluate on test set
        print("\nEvaluating classifier on test set:")
        classifier.evaluate(X_test, y_test)

        # HAKIM's component: Save the models
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save PCA model
        with open(os.path.join(model_dir, "pca_model.pkl"), "wb") as f:
            pickle.dump(pca_model, f)
            
        # Save classifier data
        with open(os.path.join(model_dir, "classifier.pkl"), "wb") as f:
            pickle.dump({'X_train': X_train, 'y_train': y_train}, f)
            
        # Save label mapping
        with open(os.path.join(model_dir, "label_mapping.pkl"), "wb") as f:
            pickle.dump(label_to_name, f)
            
        print(f"Models saved to '{model_dir}' directory")

        # ALTAIR's component: Run real-time recognition
        print("\nStarting real-time recognition...")
        recognition = RealTimeRecognition(pca_model, classifier, label_to_name)
        recognition.run()

    except Exception as e:
        print(f"An error occurred: {e}")


# Entry point of application
if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        main()
    except ImportError:
        print("Please install matplotlib to visualize PCA components:")
        print("pip install matplotlib")
        print("Running without visualization...")
        main()