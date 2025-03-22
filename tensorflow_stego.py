
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
import numpy as np
import cv2
from cryptography.fernet import Fernet
import base64
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import io


def derive_key(user_key):
    """Expands a 4-character user key into a 32-byte encryption key using PBKDF2."""
    salt = b"stegano_salt"  # A fixed salt (can be randomized for more security)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    return base64.urlsafe_b64encode(kdf.derive(user_key.encode()))


def encrypt_data(data, key):
    """Encrypts data using the given key."""
    cipher = Fernet(key)
    return cipher.encrypt(data)


def decrypt_data(encrypted_data, key):
    """Decrypts encrypted data using the given key."""
    cipher = Fernet(key)
    return cipher.decrypt(encrypted_data)


def build_unet():
    """Builds a simple U-Net architecture (not used in current embedding but can be enhanced)."""
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(inputs, decoded)


def embed_file_in_image(image_bytes, file_bytes, key):
    """Embeds an encrypted file into an image using LSB steganography."""
    # Read image from bytes
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read the image.")
    
    image = cv2.resize(image, (256, 256))  # Resize image to 256x256

    # Encrypt the file data
    encrypted_data = encrypt_data(file_bytes, key)
    binary_data = np.unpackbits(np.frombuffer(encrypted_data, dtype=np.uint8))

    # Flatten image for modification
    flat_image = image.flatten().astype(np.uint8)
    bits_to_embed = min(len(binary_data), len(flat_image))

    # Embed bits into LSB
    for i in range(bits_to_embed):
        flat_image[i] = (flat_image[i] & 0xFE) | binary_data[i]

    stego_image = flat_image.reshape((256, 256, 3))

    # Convert image to bytes
    _, buffer = cv2.imencode('.png', stego_image)
    return buffer.tobytes(), len(binary_data)


def extract_file_from_image(stego_image_bytes, key, bits_count=None):
    """Extracts and decrypts a hidden file from a steganographic image."""
    # Read image from bytes
    image_np = np.frombuffer(stego_image_bytes, np.uint8)
    stego_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if stego_image is None:
        raise ValueError("Could not read the stego image.")

    stego_image = cv2.resize(stego_image, (256, 256))

    # Flatten image to extract bits
    flat_image = stego_image.flatten()

    # Extract bits - use all bits if bits_count is None, otherwise use specified amount
    if bits_count is None:
        bits_count = len(flat_image)
    
    extracted_bits = []
    for i in range(min(bits_count, len(flat_image))):
        extracted_bits.append(flat_image[i] & 1)  # Get LSB

    extracted_bits = np.array(extracted_bits, dtype=np.uint8)
    extracted_data = np.packbits(extracted_bits)

    # Decrypt the data
    try:
        decrypted_data = decrypt_data(extracted_data.tobytes(), key)
        return decrypted_data
    except Exception as e:
        raise ValueError(f"Error during decryption: {e}")
