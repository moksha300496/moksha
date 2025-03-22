import streamlit as st
import io
from utils.crypto import CryptoHandler
from utils.steganography import Steganography
from utils.unet import DeepSteganography
from utils.tensorflow_stego import derive_key, embed_file_in_image, extract_file_from_image
import base64
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def get_download_link(file_bytes, filename, text):
    """Generate a download link for bytes data"""
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'

@st.cache_resource
def load_deep_stego_model():
    """Load and cache the deep steganography model"""
    return DeepSteganography()

def prepare_image_for_unet(image_bytes):
    """Convert image bytes to tensor for U-Net"""
    image = Image.open(io.BytesIO(image_bytes))
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def tensor_to_bytes(tensor, format='PNG'):
    """Convert tensor to bytes"""
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

def main():
    st.set_page_config(
        page_title="Advanced Steganography",
        page_icon="🔒",
        layout="wide"
    )

    st.title("🔒 Advanced Steganography Application")
    st.markdown("""
    This application allows you to:
    * Hide files within images using advanced deep learning (U-Net/GAN) or traditional methods
    * Files are automatically encrypted for security
    * Extract and decrypt hidden files
    """)

    # Load U-Net model
    deep_stego = load_deep_stego_model()

    operation = st.radio(
        "Select Operation",
        ["Hide File", "Extract File"]
    )
    
    # Add a selectbox for choosing the steganography method
    stego_method = st.selectbox(
        "Select Steganography Method",
        ["U-Net/GAN (Deep Learning)", "LSB (Traditional)", "TensorFlow LSB"]
    )

    if operation == "Hide File":
        st.header("Hide File in Image")

        # File uploads
        carrier_image = st.file_uploader("Upload Carrier Image (PNG/JPEG)", type=['png', 'jpg', 'jpeg'])
        secret_file = st.file_uploader("Upload File to Hide", type=['txt', 'pdf', 'doc', 'docx'])

        method = st.selectbox(
            "Select Steganography Method",
            ["U-Net/GAN (Deep Learning)", "LSB (Traditional)"]
        )

        if carrier_image and secret_file:
            if st.button("Process"):
                try:
                    with st.spinner("Processing..."):
                        # Read files
                        image_bytes = carrier_image.getvalue()
                        secret_bytes = secret_file.getvalue()

                        # Always encrypt data for security
                        key = CryptoHandler.generate_key()
                        encrypted_data = CryptoHandler.encrypt_data(secret_bytes, key)

                        # Provide key download
                        st.markdown(
                            get_download_link(key, "encryption_key.key", "📥 Download Encryption Key"),
                            unsafe_allow_html=True
                        )
                        st.warning("⚠️ Keep this key safe! You'll need it to extract the file later.")

                        # Perform steganography based on selected method
                        if method == "U-Net/GAN (Deep Learning)":
                            # Prepare image for U-Net
                            image_tensor = prepare_image_for_unet(image_bytes)

                            # Prepare secret data: take first 3 bytes and normalize to [-1, 1]
                            secret_data = np.frombuffer(encrypted_data[:3], dtype=np.uint8)
                            # Ensure we have exactly 3 values for RGB
                            if len(secret_data) < 3:
                                secret_data = np.pad(secret_data, (0, 3 - len(secret_data)))
                            else:
                                secret_data = secret_data[:3]

                            # Convert to tensor and reshape for batch processing
                            secret_tensor = torch.from_numpy(secret_data).float()
                            secret_tensor = (secret_tensor / 127.5) - 1
                            secret_tensor = secret_tensor.to(deep_stego.device)
                            secret_tensor = secret_tensor.view(1, 3)  # Reshape to (batch_size, channels)

                            # Generate stego image
                            stego_tensor = deep_stego.encode_image(image_tensor, secret_tensor)
                            result_bytes = tensor_to_bytes(stego_tensor)

                            # Store the rest of encrypted data as metadata
                            metadata = encrypted_data[3:]
                            st.markdown(
                                get_download_link(metadata, "metadata.bin", "📥 Download Additional Data"),
                                unsafe_allow_html=True
                            )
                            st.info("⚠️ You'll need both the image and the additional data file for extraction!")
                        elif method == "TensorFlow LSB":
                            # Use TensorFlow LSB method
                            tf_key = st.text_input("Enter a 4-character secret key for encryption:", max_chars=4)
                            if tf_key and len(tf_key) == 4:
                                derived_key = derive_key(tf_key)
                                result_bytes, bits_count = embed_file_in_image(image_bytes, secret_bytes, derived_key)
                                
                                # Store bits count as metadata
                                metadata = str(bits_count).encode()
                                st.markdown(
                                    get_download_link(metadata, "bits_count.txt", "📥 Download Bits Count"),
                                    unsafe_allow_html=True
                                )
                                st.info("⚠️ You'll need to keep track of your 4-character key and the bits count for extraction!")
                            else:
                                st.error("Please enter a 4-character secret key")
                                return
                        else:
                            # Use traditional LSB method
                            result_bytes = Steganography.encode(image_bytes, encrypted_data)

                        # Provide result download
                        st.markdown(
                            get_download_link(result_bytes, "hidden_file.png", "📥 Download Result Image"),
                            unsafe_allow_html=True
                        )
                        st.success("✅ File successfully hidden!")

                except ValueError as e:
                    st.error(f"Error: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

    else:
        st.header("Extract Hidden File")

        # File uploads
        stego_image = st.file_uploader("Upload Image with Hidden Data", type=['png', 'jpg', 'jpeg'])
        method = st.selectbox(
            "Select Steganography Method",
            ["U-Net/GAN (Deep Learning)", "LSB (Traditional)"]
        )

        if method == "U-Net/GAN (Deep Learning)":
            metadata_file = st.file_uploader("Upload Additional Data File", type=['bin'])
        elif method == "TensorFlow LSB":
            tf_key = st.text_input("Enter your 4-character decryption key:", max_chars=4)
            bits_count_file = st.file_uploader("Upload Bits Count File", type=['txt'])

        key_file = st.file_uploader("Upload Encryption Key", type=['key']) if method != "TensorFlow LSB" else None

        if stego_image and ((method == "TensorFlow LSB" and tf_key and bits_count_file) or 
                           (method == "U-Net/GAN (Deep Learning)" and key_file and metadata_file) or
                           (method == "LSB (Traditional)" and key_file)):
            if st.button("Extract"):
                try:
                    with st.spinner("Extracting..."):
                        if method == "U-Net/GAN (Deep Learning)":
                            # Extract using U-Net
                            image_tensor = prepare_image_for_unet(stego_image.getvalue())
                            extracted_tensor = deep_stego.decode_image(image_tensor)

                            # Convert tensor back to bytes
                            extracted_tensor = extracted_tensor.squeeze(0)
                            extracted_tensor = ((extracted_tensor + 1) * 127.5).byte()
                            extracted_part = extracted_tensor.cpu().numpy().tobytes()

                            # Combine with metadata
                            extracted_data = extracted_part + metadata_file.getvalue()
                        elif method == "TensorFlow LSB":
                            # Read bits count from file
                            bits_count = int(bits_count_file.getvalue().decode().strip())
                            
                            # Derive key from user input
                            derived_key = derive_key(tf_key)
                            
                            # Extract using TensorFlow LSB
                            decrypted_data = extract_file_from_image(stego_image.getvalue(), derived_key, bits_count)
                            
                            # Provide extracted file download
                            st.markdown(
                                get_download_link(decrypted_data, "extracted_file", "📥 Download Extracted File"),
                                unsafe_allow_html=True
                            )
                            st.success("✅ File successfully extracted!")
                            return
                        else:
                            # Extract using LSB
                            extracted_data = Steganography.decode(stego_image.getvalue())

                        # Decrypt data
                        key = key_file.getvalue()
                        decrypted_data = CryptoHandler.decrypt_data(extracted_data, key)

                        # Provide extracted file download
                        st.markdown(
                            get_download_link(decrypted_data, "extracted_file", "📥 Download Extracted File"),
                            unsafe_allow_html=True
                        )
                        st.success("✅ File successfully extracted!")

                except ValueError as e:
                    st.error(f"Error: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()