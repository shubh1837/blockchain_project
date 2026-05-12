import time
import requests
import uuid

class BlockchainClient:
    def __init__(self, api_endpoint="http://127.0.0.1:8080/api"):
        """
        Initializes the connection to the Hyperledger Fabric REST API wrapper.
        Includes a standalone Mock mode if the Fabric network is offline.
        """
        self.api_endpoint = api_endpoint
        self.is_connected = self._check_connection()

    def _check_connection(self):
        try:
            response = requests.get(f"{self.api_endpoint}/health", timeout=2)
            if response.status_code == 200:
                print("Blockchain Hyperledger Ledger Connected.")
                return True
        except requests.exceptions.RequestException:
            pass
        print("Warning: Blockchain Network not detected. Using Standalone MOCK Ledger mode.")
        return False

    def log_diagnostic_record(self, cid, ssh_hash, patient_id="AUTH_ANON_99X"):
        """
        Pushes the IPFS CID and SSH Hash to the immutable ledger.
        """
        import sys
        node_port = "8000"
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                node_port = sys.argv[i+1]

        record_id = f"REC_{uuid.uuid4().hex[:8].upper()}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        uploader_id = f"HOSPITAL_NODE_{node_port}"
        
        payload = {
            "recordID": record_id,
            "patientID": patient_id,
            "ipfsCID": cid,
            "sshHash": ssh_hash,
            "timestamp": timestamp,
            "uploaderID": uploader_id
        }

        if not self.is_connected:
            return self._mock_submit(payload)
            
        try:
            # Assuming a REST API middleware mapping to CreateDiagnosticRecord
            response = requests.post(f"{self.api_endpoint}/invoke/CreateDiagnosticRecord", json=payload)
            response.raise_for_status()
            print(f"Transaction Committed to Blockchain: {record_id}")
            return record_id
        except requests.exceptions.RequestException as e:
            print(f"Blockchain submission failed: {e}")
            return self._mock_submit(payload)

    def _mock_submit(self, payload):
        """Mock transaction logger for pipeline testing."""
        print(f"\n[BLOCKCHAIN MOCK] Writing block to local state...")
        print(f"  -> TX_ID: {payload['recordID']}")
        print(f"  -> CID_ATTACHED: {payload['ipfsCID']}")
        print(f"  -> SSH_INTEGRITY: {payload['sshHash']}")
        print(("[BLOCKCHAIN MOCK] Transaction Successfully Committed.\n"))
        return payload['recordID']

if __name__ == "__main__":
    bc = BlockchainClient()
    bc.log_diagnostic_record("QmTestHash123", "P:ff88-C:abcd")
