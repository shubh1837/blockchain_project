# Chapter 3: Methodology

## 3.1 Introduction
This chapter enumerates the overarching technological methodology constructed to address the "Data Silo" paradox in modern radiological AI. The goal of the proposed architecture is to permit cross-institutional Deep Learning training without compromising patient confidentiality under HIPAA regulations.

## 3.2 Subject Sensitive Hashing (SSH) Layer
Standard cryptographic hashes (like SHA-256) are hyper-sensitive; altering a single pixel fundamentally recalculates the exact identity. In radiology, image compression artifacts frequently occur across hospital environments. A strict SHA-256 validation would mark compressed but otherwise diagnostically valid X-rays as "tampered."

To rectify this, we introduced the concept of the **Subject Sensitive Hash (SSH)**, implementing a structural `dHash` algorithm. The dHash compares gradients between adjacent pixels. By fusing this perceptual hash with a cryptographic standard (`P:<perceptual>-C:<crypto>`), the smart contract can mathematically vet whether the visual diagnostic semantics of the image have been maliciously modified by an attacker, or legally compressed by a clinician.

## 3.3 The Decentralized Storage (IPFS) Protocol
Given that medical DICOM files and high-resolution PNGs often exceed standard transactional limits across blockchain ledgers, all unstructured raw data is inherently stored off-chain.

The `Kubo` InterPlanetary File System (IPFS) coordinates peer-to-peer distribution. Rather than maintaining centralized databases, participating hospitals push the structured matrices onto the IPFS swarm. This yields a deterministic `CID` (Content Identifier) acting as an absolute network-agnostic pointer.

## 3.4 Hyperledger Fabric Orchestration
At the core of the immutability logic lies a Hyperledger Fabric implementation. Written in Golang, the `xray_contract.go` chaincode dictates the exact format of a legitimate transaction block. The global state manages:
1. Anonymized `PatientID`
2. The exact IPFS pointer `CID`
3. The SSH verification code

By distributing this ledger across clinical organizations, no single hospital can monopolize or secretly manipulate the patient mappings.

## 3.5 Federated Learning Strategy (Flower Framework)
Rather than aggregating 42 Gigabytes of sensitive imaging onto a centralized server, the algorithm brings the *model to the data*.

Deployed via `flwr`, the global master server emits the `DenseNet121` baseline weights to connecting hospital client nodes. These local clients execute a PyTorch optimizer loop (`BCEWithLogitsLoss`) exclusively on their local data subset, computing the delta gradients. Only the raw weight parameters are transmitted back to the global aggregator utilizing the `FedAvg` (Federated Averaging) strategy.
