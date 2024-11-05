import numpy as np
import cv2
from scipy.fftpack import dct
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """A class for detecting deepfake images using computer vision and machine learning techniques."""

    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the DeepfakeDetector.

        Args:
            n_estimators (int): Number of trees in the random forest
            random_state (int): Random seed for reproducibility
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.is_trained = False

    def extract_features(self, image_path):
        """
        Extract features from an image for deepfake detection.

        Args:
            image_path (str): Path to the image file

        Returns:
            np.array: Array of extracted features
        """
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Convert to grayscale
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise ValueError(f"Error converting image to grayscale: {str(e)}")

        # Extract features
        features = []

        # 1. Extract noise features
        noise_features = self._extract_noise_features(gray)
        features.extend(noise_features)

        # 2. Analyze compression artifacts
        compression_features = self._analyze_compression(gray)
        features.extend(compression_features)

        # 3. Extract color statistics
        color_features = self._analyze_color_statistics(img)
        features.extend(color_features)

        return np.array(features)

    def _extract_noise_features(self, gray_img):
        """Extract noise-related features from grayscale image."""
        try:
            denoised = cv2.medianBlur(gray_img, 3)
            noise = cv2.absdiff(gray_img, denoised)

            return [
                np.mean(noise),
                np.std(noise),
                np.max(noise),
                np.percentile(noise, 90)
            ]
        except Exception as e:
            raise ValueError(f"Error extracting noise features: {str(e)}")

    def _analyze_compression(self, gray_img):
        """Analyze JPEG compression artifacts."""
        try:
            dct_array = dct(dct(gray_img.astype(float), axis=0), axis=1)

            return [
                np.mean(np.abs(dct_array)),
                np.std(dct_array),
                np.percentile(np.abs(dct_array), 90),
                np.sum(np.abs(dct_array) > 0.1) / dct_array.size
            ]
        except Exception as e:
            raise ValueError(f"Error analyzing compression: {str(e)}")

    def _analyze_color_statistics(self, img):
        """Extract color-based statistical features."""
        try:
            features = []
            for channel in cv2.split(img):
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.percentile(channel, 10),
                    np.percentile(channel, 90)
                ])
            return features
        except Exception as e:
            raise ValueError(f"Error analyzing color statistics: {str(e)}")

    def train(self, real_dir, fake_dir):
        """
        Train the detector on a dataset of real and fake images.

        Args:
            real_dir (str): Directory containing real images
            fake_dir (str): Directory containing fake images

        Returns:
            float: Model accuracy on test set
        """
        # Validate directories
        if not os.path.isdir(real_dir):
            raise NotADirectoryError(f"Real images directory not found: {real_dir}")
        if not os.path.isdir(fake_dir):
            raise NotADirectoryError(f"Fake images directory not found: {fake_dir}")

        features = []
        labels = []

        # Process real images
        real_images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not real_images:
            raise ValueError(f"No valid images found in real directory: {real_dir}")

        logger.info(f"Processing {len(real_images)} real images...")
        for img_name in real_images:
            try:
                features.append(self.extract_features(os.path.join(real_dir, img_name)))
                labels.append(0)  # 0 for real
            except Exception as e:
                logger.warning(f"Error processing real image {img_name}: {str(e)}")

        # Process fake images
        fake_images = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not fake_images:
            raise ValueError(f"No valid images found in fake directory: {fake_dir}")

        logger.info(f"Processing {len(fake_images)} fake images...")
        for img_name in fake_images:
            try:
                features.append(self.extract_features(os.path.join(fake_dir, img_name)))
                labels.append(1)  # 1 for fake
            except Exception as e:
                logger.warning(f"Error processing fake image {img_name}: {str(e)}")

        if not features:
            raise ValueError("No features extracted from any images")

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Calculate accuracy
        accuracy = self.model.score(X_test, y_test)
        logger.info(f"Training completed. Test accuracy: {accuracy:.2f}")

        return accuracy

    def predict(self, image_path):
        """
        Predict if an image is real or fake.

        Args:
            image_path (str): Path to the image file

        Returns:
            dict: Prediction results including classification and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained. Please train the model first.")

        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Extract features and predict
        try:
            features = self.extract_features(image_path)
            prediction = self.model.predict([features])[0]
            probability = self.model.predict_proba([features])[0]

            return {
                'is_fake': bool(prediction),
                'confidence': float(probability[prediction]),
                'probabilities': {
                    'real': float(probability[0]),
                    'fake': float(probability[1])
                }
            }
        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['Real_Image', 'Fake_Image', 'test_Image']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")


def main():
    """Main function to run the deepfake detector."""
    try:
        # Create directories
        setup_directories()

        # Initialize detector
        detector = DeepfakeDetector()
        logger.info("Initialized DeepfakeDetector")

        # Check for training data
        real_dir = 'Real_Image'
        fake_dir = 'Fake_Image'

        real_images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        fake_images = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(real_images) < 5 or len(fake_images) < 5:
            logger.warning(f"""
            Insufficient training data:
            Real images found: {len(real_images)}
            Fake images found: {len(fake_images)}
            Please add at least 5 images to each directory for minimal training.
            """)
            return

        logger.info(f"Found {len(real_images)} real images and {len(fake_images)} fake images")

        # Train the model
        accuracy = detector.train(real_dir, fake_dir)
        logger.info(f"Model trained with accuracy: {accuracy:.2f}")

        # Make prediction on test image
        test_image = 'test_Image/fake_9966.jpg'
        if os.path.exists(test_image):
            result = detector.predict(test_image)
            logger.info(f"""
            Prediction Results:
            Classification: {'Fake' if result['is_fake'] else 'Real'}
            Confidence: {result['confidence']:.2%}
            Probability distribution:
                Real: {result['probabilities']['real']:.2%}
                Fake: {result['probabilities']['fake']:.2%}
            """)
        else:
            logger.warning(f"No test image found at {test_image}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()