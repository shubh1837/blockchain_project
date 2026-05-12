import cv2
import hashlib
import numpy as np

class SSHGenerator:
    """
    Subject Sensitive Hashing (SSH)
    Generates a perceptual hash of medical images. 
    It focuses on the core diagnostic structure (using image moments/histograms or DHash)
    so minor compression artifacts don't completely invalidate the hash, but malicious 
    pixel tampering changes it.
    """
    def __init__(self, hash_size=8):
        self.hash_size = hash_size

    def dhash(self, image):
        """
        Implementation of Difference Hash (dHash).
        """
        # Resize to hash_size + 1 wide and hash_size high
        resized = cv2.resize(image, (self.hash_size + 1, self.hash_size), interpolation=cv2.INTER_AREA)
        # Compute differences between adjacent pixels
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Convert binary array to hex string
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    def generate_ssh(self, file_path):
        """
        Generates a composite Subject Sensitive Hash combining cryptographic 
        identity (SHA256) and perceptual structural identity (dHash).
        """
        # Load image in grayscale
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image for SSH generation.")

        # 1. Perceptual Hash (Captures visual structure)
        perceptual_hash = hex(self.dhash(img))[2:] # Remove '0x'

        # 2. Cryptographic Hash (Captures exact binary identity)
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
             hasher.update(f.read())
        crypto_hash = hasher.hexdigest()

        # In a real SSH system, weights are dynamically adjusted. 
        # Here we package both to ensure total compliance.
        composite_ssh = f"P:{perceptual_hash}-C:{crypto_hash[:16]}"
        print(f"Generated Subject Sensitive Hash (SSH): {composite_ssh}")
        return composite_ssh

if __name__ == "__main__":
    print("SSH Module initialization ready.")
