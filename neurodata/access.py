"""
Access control: AES-GCM encryption on .kess files.
Only someone with the correct password can load the model.
"""
import os
import io
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200_000)
    return kdf.derive(password.encode())


def save_locked(checkpoint: dict, path: str, password: str):
    """Encrypt and save a checkpoint. Only loadable with the correct password."""
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    plaintext = buf.getvalue()

    salt  = os.urandom(16)
    nonce = os.urandom(12)
    key   = _derive_key(password, salt)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)

    with open(path, 'wb') as f:
        f.write(salt + nonce + ciphertext)

    print(f"Model saved (encrypted) → {path}")


def load_locked(path: str, password: str) -> dict:
    """Decrypt and load a checkpoint. Raises ValueError on wrong password."""
    with open(path, 'rb') as f:
        data = f.read()

    salt, nonce, ciphertext = data[:16], data[16:28], data[28:]
    key = _derive_key(password, salt)
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    except Exception:
        raise ValueError("Access denied: incorrect password.")

    buf = io.BytesIO(plaintext)
    return torch.load(buf, map_location='cpu', weights_only=False)
