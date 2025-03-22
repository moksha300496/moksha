from cryptography.fernet import Fernet
import base64
import os

class CryptoHandler:
    @staticmethod
    def generate_key():
        """Generate a new encryption key"""
        return Fernet.generate_key()

    @staticmethod
    def encrypt_data(data: bytes, key: bytes) -> bytes:
        """Encrypt data using Fernet symmetric encryption"""
        if not isinstance(key, bytes):
            key = key.encode()
        
        f = Fernet(key)
        return f.encrypt(data)

    @staticmethod
    def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using Fernet symmetric encryption"""
        if not isinstance(key, bytes):
            key = key.encode()
            
        try:
            f = Fernet(key)
            return f.decrypt(encrypted_data)
        except Exception as e:
            raise ValueError("Invalid key or corrupted data")
