from PIL import Image
import numpy as np
import io

class Steganography:
    @staticmethod
    def to_binary(data):
        """Convert data to binary format"""
        if isinstance(data, str):
            return ''.join([format(ord(i), "08b") for i in data])
        elif isinstance(data, bytes):
            return ''.join([format(i, "08b") for i in data])
        elif isinstance(data, np.ndarray):
            return [format(i, "08b") for i in data]
        else:
            raise TypeError("Type not supported")

    @staticmethod
    def encode(image_bytes: bytes, secret_data: bytes) -> bytes:
        """Encode secret data into image using LSB steganography"""
        # Convert image bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert image to numpy array
        image_array = np.array(image)
        
        # Calculate maximum bytes that can be encoded
        n_bytes = image_array.shape[0] * image_array.shape[1] * 3 // 8
        if len(secret_data) > n_bytes:
            raise ValueError("Error: Insufficient bytes, need bigger image or less data!")

        # Add length of data to the beginning of secret_data
        secret = len(secret_data).to_bytes(4, 'big') + secret_data
        
        binary_secret = Steganography.to_binary(secret)
        data_index = 0
        
        # Flatten the image
        modified_pixels = image_array.reshape(-1)
        
        # Modify the least significant bits
        for i in range(0, len(binary_secret), 1):
            modified_pixels[i] = (modified_pixels[i] & 254) | int(binary_secret[i])
        
        # Reshape back to original shape
        modified_image = modified_pixels.reshape(image_array.shape)
        
        # Convert back to image
        result = Image.fromarray(modified_image)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format=image.format)
        return img_byte_arr.getvalue()

    @staticmethod
    def decode(image_bytes: bytes) -> bytes:
        """Decode secret data from image"""
        # Convert image bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert image to numpy array
        image_array = np.array(image)
        
        # Flatten the image
        pixels = image_array.reshape(-1)
        
        # Extract the LSB of each pixel
        binary_data = ''.join([str(pixel & 1) for pixel in pixels])
        
        # First 32 bits contain the length of the data
        length_bits = binary_data[:32]
        length = int(length_bits, 2)
        
        # Extract the actual data
        data_bits = binary_data[32:32+(length*8)]
        
        # Convert bits to bytes
        data_bytes = []
        for i in range(0, len(data_bits), 8):
            byte = data_bits[i:i+8]
            data_bytes.append(int(byte, 2))
            
        return bytes(data_bytes)
