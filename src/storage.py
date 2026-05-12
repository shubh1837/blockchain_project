import os
import requests
import hashlib

class IPFSStorage:
    def __init__(self, api_url="http://127.0.0.1:5001/api/v0"):
        """
        Initializes the IPFS storage interface.
        If a local daemon isn't running, it will fall back to returning a mock hash.
        """
        self.api_url = api_url
        self.is_connected = self._check_connection()

    def _check_connection(self):
        try:
            # Ping IPFS daemon to check if it's alive
            response = requests.post(f"{self.api_url}/id", timeout=2)
            if response.status_code == 200:
                print("IPFS Daemon Connected Successfully.")
                return True
        except requests.exceptions.RequestException:
            pass
        print("Warning: IPFS Daemon not detected at 127.0.0.1:5001. Using MOCK storage mode.")
        return False

    def upload_to_ipfs(self, file_path):
        """
        Uploads a file to IPFS and returns the CID.
        Returns a mock CID if daemon is unreachable.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.is_connected:
            return self._mock_upload(file_path)

        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    f"{self.api_url}/add",
                    files={"file": f}
                )
            response.raise_for_status()
            cid = response.json()["Hash"]
            print(f"File uploaded to IPFS with CID: {cid}")
            return cid
        except requests.exceptions.RequestException as e:
            print(f"Failed to upload to IPFS: {e}")
            return self._mock_upload(file_path)

    def download_from_ipfs(self, cid, output_path):
        """
        Downloads a file from IPFS by CID to output_path.
        Returns mock confirmation if daemon is unreachable.
        """
        if not self.is_connected:
            print(f"Mocking download of CID: {cid} to {output_path}")
            return

        try:
            response = requests.post(
                f"{self.api_url}/cat?arg={cid}"
            )
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"File downloaded successfully to {output_path}")
        except requests.exceptions.RequestException as e:
             print(f"Failed to download from IPFS: {e}")

    def _mock_upload(self, file_path):
        """Generates a pseudo IPFS CID hash for offline testing."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
             buf = f.read(65536)
             while len(buf) > 0:
                 hasher.update(buf)
                 buf = f.read(65536)
        
        # Format as a pseudo IPFS v0 Content ID (Qm...)
        mock_cid = "QmMock" + hasher.hexdigest()[:42]
        print(f"File MOCK-uploaded. Generated Mock CID: {mock_cid}")
        return mock_cid

if __name__ == "__main__":
    storage = IPFSStorage()
    print(f"Storage connection active: {storage.is_connected}")
