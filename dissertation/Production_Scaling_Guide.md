# Production Scaling Guide: DacNet Real-World Implementation

This document outlines the transition of the DacNet (Decentralized AI Clinical Network) architecture from a local, single-machine simulation into a highly distributed, enterprise-grade production environment across real hospitals.

## Phase 1: Storage and Infrastructure Setup 🗄️

Currently, the IPFS client and file storage are simulated locally.
* **Production IPFS Nodes:** Instead of a mock client, actual IPFS nodes (`Kubo`) must run using the provided `docker-compose-ipfs.yml`. The system will establish a **private IPFS network (swarm)** restricted only to participating hospital IPs, preventing public internet access to the Content Identifiers (CIDs).
* **Redundancy (IPFS Cluster):** To ensure high availability, an IPFS Cluster will be implemented. If Hospital A's systems go offline, Hospital B and C will maintain pinned backups of the encrypted X-rays.
* **Traditional Database:** Standard relational databases (e.g., PostgreSQL) will be maintained beside the FastAPI backend to store normal operational data (e.g., Doctor login credentials, UI states) that does not require the immutability of a blockchain.

## Phase 2: Enterprise Blockchain Network (Hyperledger Fabric) ⛓️

The current `blockchain_client.py` simulates a ledger. In reality, patient metadata cannot be fully public, ruling out public blockchains.
* **Deploy Hyperledger Fabric**: An enterprise permissioned blockchain must be deployed using the Go smart contracts (`blockchain/chaincode/xray_contract.go`).
* **Network Consortium**: A consortium consisting of hospitals and regulatory bodies will be established. Each entity controls an "Orderer Node" to ensure decentralized consensus, preventing any single hospital from dominating the network or rewriting history.
* **Identity Management**: Integration with Hyperledger Fabric CA (Certificate Authority) is necessary so every medical professional receives a cryptographic X.509 certificate to sign their uploads definitively.

## Phase 3: Federated Learning Hardware & Orchestration 🤖

The simulated system runs `fl_server.py` and `fl_client.py` on a single localized GPU/CPU.
* **The FL Aggregator (AWS/GCP)**: The central `fl_server.py` must be deployed on a highly-available cloud instance with sufficient bandwidth to accept gigabytes of model weight updates simultaneously from participating nodes.
* **Hospital Edge Hardware**: Every participating hospital will install on-premise GPU servers (e.g., NVIDIA DGX computing systems). The `fl_client.py` will run natively inside the hospital's secured firewall on local private data.
* **Firewall Configuration**: Hospital IT administrations will configure outbound port rules allowing the `fl_client.py` to transmit weight updates to the central `fl_server.py`, strictly limiting inbound ports to combat hacking vulnerabilities.

## Phase 4: Application Deployment & Scalability 🌐

* **Containerization**: All primary components (Frontend React dashboard, FastAPI backend, Flower ML Client) must be packaged into isolated **Docker containers**.
* **Kubernetes (K8s)**: Kubernetes will orchestrate these containers. In the event of high diagnostic volume traffic (e.g., a regional health crisis), Kubernetes will automatically dynamically scale the FastAPI pods to handle the load without crashing the service.
* **Reverse Proxies & SSL**: Reverse proxies like NGINX or Traefik will be deployed to enforce SSL/TLS certificates (HTTPS routing), ensuring that doctors' and administrators' connections to the dashboard are perfectly encrypted.

## Phase 5: Legal, Security, and Compliance ⚖️

A purely technical implementation will falter without thorough legal compliance regarding healthcare data security laws.
* **HIPAA / GDPR Compliance**: Although Federated Learning mitigates the sharing of raw patient data, it must be mathematically and legally proven that individual *model weights* cannot be reverse-engineered to expose patient identities via model-inversion attacks.
* **End-to-End Encryption (E2EE)**: Before any X-ray is submitted to the private IPFS network, it must be robustly encrypted using AES-256 protocols. The IPFS CID will only return scrambled ciphertext unless the requesting doctor holds the respective decryption key.
* **BAA Agreements**: Rigorous Business Associate Agreements (BAAs) must be established between all participating hospitals, cloud providers, and IT management personnel.

---

### Initial Setup Checklist for DevOps
1. Containerize the `frontend` and `api_server` via `Dockerfile`.
2. Boot the `Kubo` IPFS daemon and configure `storage.py` to communicate directly via the HTTP API rather than mocked local storage.
3. Deploy the Golang smart contract to a cloud testing network and interconnect the Python application using the Hyperledger Fabric Python SDK.
