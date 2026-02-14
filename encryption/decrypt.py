import sys
import os
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet

load_dotenv()


def decrypt_file_content(encrypted_file_path):
    private_key_pem = os.environ.get("SUBMISSION_PRIVATE_KEY")

    if not private_key_pem:
        raise ValueError("Error: 'SUBMISSION_PRIVATE_KEY' missing from environment.")

    # --------- FIX 1: Handle newline formatting properly ----------
    private_key_pem = private_key_pem.replace('\\n', '\n').strip()

    try:
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=None
        )
    except Exception as e:
        raise ValueError(f"Invalid Private Key format: {e}")

    if not os.path.exists(encrypted_file_path):
        raise FileNotFoundError(f"File not found: {encrypted_file_path}")

    with open(encrypted_file_path, "rb") as f:
        file_content = f.read()

    # --------- FIX 2: Dynamic RSA block size ----------
    rsa_segment_size = private_key.key_size // 8

    if len(file_content) < rsa_segment_size:
        raise ValueError("Encrypted file too short to contain valid RSA header.")

    encrypted_session_key = file_content[:rsa_segment_size]
    encrypted_data = file_content[rsa_segment_size:]

    # --------- Decrypt session key ----------
    try:
        session_key = private_key.decrypt(
            encrypted_session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    except Exception as e:
        raise ValueError(f"RSA Decryption failed: {e}")

    # --------- Decrypt data ----------
    try:
        cipher_suite = Fernet(session_key)
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        return decrypted_data
    except Exception as e:
        raise ValueError(f"Data Decryption failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decrypt.py <filename>")
    else:
        try:
            encrypted_file = sys.argv[1]
            file_content = decrypt_file_content(encrypted_file)

            # --------- FIX 3: Avoid overwriting original accidentally ----------
            new_file_name = encrypted_file.replace(".enc", ".decrypted.csv")

            with open(new_file_name, "wb") as f:
                f.write(file_content)

            print(f"Decryption successful! Saved to '{new_file_name}'")

        except Exception as e:
            print(f"FAILED: {e}")
            sys.exit(1)
