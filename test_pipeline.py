import numpy as np
import cv2
from src.preprocessing import load_and_preprocess_image
from src.model import XRayModel
from src.storage import IPFSStorage
from src.ssh_security import SSHGenerator
from src.blockchain_client import BlockchainClient

def test_pipeline():
    print("Testing pipeline end-to-end...")
    
    # 1. Create a dummy test image (simulate an X-ray)
    dummy_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    cv2.imwrite("dummy_xray.png", dummy_img)
    
    try:
        # Phase 2: Preprocess & AI Inference
        print("\n--- PHASE 2: Local Preprocessing & AI ---")
        preprocessed_img = load_and_preprocess_image("dummy_xray.png")
        print(f"Preprocessed shape: {preprocessed_img.shape}")
        
        model = XRayModel()
        predictions = model.predict(preprocessed_img)
        print("Top 3 Predictions on dummy image:")
        for idx, (pathology, prob) in enumerate(list(predictions.items())[:3]):
             print(f"  {idx+1}. {pathology}: {prob:.4f}")
             
        # Phase 4: SSH Integrity Generation (Doing this before IPFS upload so we hash local file)
        print("\n--- PHASE 4: Integrity Verification (SSH) ---")
        ssh_gen = SSHGenerator()
        ssh_hash = ssh_gen.generate_ssh("dummy_xray.png")
             
        # Phase 3: IPFS Decentralized Storage
        print("\n--- PHASE 3: Decentralized Storage (IPFS) ---")
        storage = IPFSStorage()
        cid = storage.upload_to_ipfs("dummy_xray.png")
        
        # Phase 5: Blockchain Orchestration
        print("\n--- PHASE 5: Blockchain Orchestration ---")
        bc = BlockchainClient()
        tx_id = bc.log_diagnostic_record(cid, ssh_hash)
        print(f"Final Pipeline Output -> Record mapped to ledger with TX: {tx_id}")
        
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        import os
        if os.path.exists("dummy_xray.png"):
            os.remove("dummy_xray.png")

if __name__ == "__main__":
    test_pipeline()
