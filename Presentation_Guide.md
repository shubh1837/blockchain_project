# DacNet 3-Laptop Presentation Guide

To present your final year project impressively using 3 laptops, you will designate one laptop as the **Central Coordinator** and the other two as **Hospital Nodes (A and B)**. This perfectly demonstrates Decentralized AI and Federated Learning across a network.

> [!IMPORTANT]
> **Prerequisites:** 
> 1. All 3 laptops **must** be connected to the exact same Wi-Fi network or mobile hotspot.
> 2. Copy the entire `blockchain_project` folder to Laptop 2 and Laptop 3. *(Note: You do NOT need the `data/NIH` mock dataset anymore because the AI trains strictly on the images you upload during the presentation!)*
> 3. Ensure you have run `npm install` inside the `frontend` folder and `pip install -r requirements.txt` in the main folder on all 3 laptops.

---

## Laptop 1: The Central Node (Aggregator & Ledger)

This laptop acts as the central Federated Learning server that aggregates the AI models from the hospitals without accessing their private data. 

**1. Start the Federated Learning Server:**
- Open a terminal in the `blockchain_project` folder.
- Run the server:
  ```bash
  python src/fl_server.py
  ```
- It will say "Starting Federated Learning Aggregation Server..." and wait for at least 1 hospital to connect. As more hospitals join, it will seamlessly integrate them into the training network!


---

## Laptop 2: Hospital Node A 

This laptop represents the first localized hospital. It processes X-Rays securely without sending the raw images to the central node.

**1. Start the Hospital Backend API:**
- Open a terminal in the `blockchain_project` folder.
- Run the FastAPI server:
  ```bash
  uvicorn src.api_server:app --host 0.0.0.0 --port 8000
  ```

**2. Start the Hospital Dashboard (Frontend):**
- Open a **second** terminal and navigate to the `frontend` folder.
- Run `npm run dev` and open `http://localhost:5173` in the browser.

**3. Connect to the Central Node for AI Training:**
- Open a **third** terminal in the `blockchain_project` folder.
- Run the Federated Learning client (it will automatically discover the central server!):
  ```bash
  python src/fl_client.py
  ```
> *Note: It will say "Dataset empty. Waiting 5s..." because you haven't uploaded an image yet. Leave it running!*

---

## Laptop 3: Hospital Node B

This laptop represents the second localized hospital, demonstrating how multiple institutions train the global AI cooperatively.

**1. Start the Hospital Backend API:**
- Open a terminal in the `blockchain_project` folder.
- Run the FastAPI server:
  ```bash
  uvicorn src.api_server:app --host 0.0.0.0 --port 8000
  ```

**2. Start the Hospital Dashboard (Frontend):**
- Open a **second** terminal and navigate to the `frontend` folder.
- Run `npm run dev` and open `http://localhost:5173` in the browser.

**3. Connect to the Central Node for AI Training:**
- Open a **third** terminal in the `blockchain_project` folder.
- Run the Federated Learning client:
  ```bash
  python src/fl_client.py
  ```
> *Note: It will also poll and wait for images.*

---

## 🎬 The Presentation Flow (What to show the panel)

Here is the best sequence to show the panel during your BTech defense:

1. **The Starting State:** Point to Laptop 2 and Laptop 3 terminals. Show the panel that `fl_client.py` is safely waiting and polling because the hospitals currently have no new patient data.
2. **Patient Uploads:** On Laptop 2 and Laptop 3, upload an X-ray image on the dashboards. Show the panel that the AI provides an initial baseline diagnosis, but the doctor has to formally verify it. Mention that the cryptographic hash/CID is generated securely for the blockchain.
3. **The Trigger:** Explain to the panel: *"By law, Hospital A cannot send its patient data to Hospital B or the Central Server. However, they both want to improve the global AI."*
4. **Active Feedback Loop:** On Laptop 2, adjust the doctor's feedback sliders and click **"Confirm Final Truth & Update AI"**. Do the exact same on Laptop 3. 
5. **The "Wow" Factor:** Instantly point to the terminals. The moment you hit confirm, Laptop 2 and 3 will detect the images and instantly join the Federated network. Show Laptop 1 aggregating the model weights across the network. Emphasize that **zero image data** ever left Laptop 2 or 3!

---

## 🔥 Bonus: Presenting Live IPFS Decentralized Storage

By default, the project falls back to a **Mock IPFS CID Generator** so the presentation works smoothly even on laptops without Docker installed. However, if you want to show the panel *actual* IPFS distributed storage working live:

**Prerequisite:** Ensure **Docker Desktop** is installed and running on Laptop 2 and Laptop 3.

1. **Start the IPFS Daemon:** Before running the backend API on Laptop 2 and Laptop 3, open a terminal in the `blockchain_project` folder and run:
   ```bash
   docker-compose -f docker-compose-ipfs.yml up -d
   ```
2. **Automatic Detection:** You don't need to change any code! When you start `uvicorn src.api_server:app` (Step 1 in Laptop 2/3), the backend will automatically scan port 5001. 
3. **The Proof:** Show the panel the backend terminal. Instead of printing `"Warning: IPFS Daemon not detected... Using MOCK storage"`, it will print `"IPFS Daemon Connected Successfully."`
4. **The Live Hash:** When you upload the X-Ray, it will push the physical file into the local offline IPFS daemon and generate a legitimate, mathematically pure `Qm...` CID hash on the dashboard, proving the decentralized storage is completely functional!
