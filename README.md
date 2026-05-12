# DacNet: Blockchain-Assisted Deep Learning Architecture for X-Ray

A decentralized medical imaging pipeline utilizing PyTorch, IPFS, Hyperledger Fabric, and Federated Learning.

## 🗂️ Project Structure
- `frontend/` - Modern React/Vite web dashboard for doctors to visualize the platform.
- `src/` - Core Python business logic (AI training, hashing, blockchains).
- `scripts/` - Utilities for bridging the dataset dependencies.
- `blockchain/` - Golang Smart Contracts designed for Hyperledger.
- `docker-compose-ipfs.yml` - DevOps file to spin up Live IPs mapping.
- `dissertation/` - Exported academic documentation and methodology summaries.

---

## 💻 1. Running the Visual Web Dashboard
We have built a premium, glassmorphism-styled UI to allow doctors/users to interact with the simulated backend pipeline seamlessly.

1. Open a terminal and navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install dependencies (First time only):
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Click the `http://localhost:5173` link in the terminal to view the UI!

---

## ⚙️ 2. Running the Pure Backend Terminal Pipeline
The main pipeline simulates the full path of a medical X-ray entering the localized hospital node, going through HIPAA-compliant AI processing, mathematically hashing for Integrity limits, and securely logging to the ledger mock.

1. Open a terminal inside the root project folder.
2. Ensure dependencies are installed (`pip install -r requirements.txt`).
3. Run the complete standalone terminal pipeline:
   ```bash
   python test_pipeline.py
   ```
*(Note for Windows users: If you encounter a console font warning during model weights download, run `$env:PYTHONIOENCODING="utf-8"; python test_pipeline.py` instead).*

---

## ⚕️ 3. Dataset Configuration (Mock vs Kaggle)
Before training deeply, the application relies on an accurate file structure for medical inputs.
```bash
python scripts/setup_nih_dataset.py
```
- **Mock Mode**: By default, this securely generates 50 synthetic localized bounds.
- **Kaggle Mode**: If you have `~/.kaggle/kaggle.json` configured on your OS, the script will automatically bypass the simulated data and programmatically download the official lightweight `nih-chest-xrays/sample` subsets directly!

---

## 🌐 4. Running Federated Learning Orchestration
To demonstrate how multiple hospital nodes can collaboratively train the overall AI model without sharing any sensitive patient data:

1. **Start the Aggregator Server**: Open a new terminal window at the project root and run:
   ```bash
   python src/fl_server.py
   ```

2. **Start Hospital Node A**: Open a *second* terminal window and run:
   ```bash
   python src/fl_client.py
   ```
   
3. **Start Hospital Node B**: Open a *third* terminal window and run:
   ```bash
   python src/fl_client.py
   ```

*(Once multiple local clients connect to the server, the Federated Averaging rounds will automatically begin simulating private multi-institutional learning).*

---

## 🚀 5. Extending to True "Live" Production
Currently, pipelines utilize a smart-mock interface to let you test the AI logic instantly without taxing your PC. 

- **Live IPFS**: Run `docker-compose -f docker-compose-ipfs.yml up -d` to launch the actual `Kubo` daemon daemonized in the background on ports `5001`/`8080`.
- **Live Hyperledger**: Deploy the smart contract located at `blockchain/chaincode/xray_contract.go` to your active Fabric deployment nodes to stop mocking block transactions.
