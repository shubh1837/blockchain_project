package main

import (
	"encoding/json"
	"fmt"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SmartContract provides functions for managing a DiagnosticRecord
type SmartContract struct {
	contractapi.Contract
}

// DiagnosticRecord describes basic details of what makes up a simple record
// It stores the IPFS link to the raw data and the SSH for integrity.
type DiagnosticRecord struct {
	RecordID   string `json:"recordID"`
	PatientID  string `json:"patientID"` // Anonymized/Hashed ID
	IPFSCID    string `json:"ipfsCID"`
	SSHHash    string `json:"sshHash"`
	Timestamp  string `json:"timestamp"`
	UploaderID string `json:"uploaderID"`
}

// InitLedger adds a base set of nodes to the ledger
func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	fmt.Println("X-Ray Storage Ledger Initialized")
	return nil
}

// CreateDiagnosticRecord issues a new diagnostic record to the world state with given details.
func (s *SmartContract) CreateDiagnosticRecord(ctx contractapi.TransactionContextInterface, recordID string, patientID string, ipfsCID string, sshHash string, timestamp string, uploaderID string) error {
	exists, err := s.RecordExists(ctx, recordID)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("the record %s already exists", recordID)
	}

	record := DiagnosticRecord{
		RecordID:   recordID,
		PatientID:  patientID,
		IPFSCID:    ipfsCID,
		SSHHash:    sshHash,
		Timestamp:  timestamp,
		UploaderID: uploaderID,
	}

	recordJSON, err := json.Marshal(record)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(recordID, recordJSON)
}

// ReadRecord returns the record stored in the world state with given id.
func (s *SmartContract) ReadRecord(ctx contractapi.TransactionContextInterface, recordID string) (*DiagnosticRecord, error) {
	recordJSON, err := ctx.GetStub().GetState(recordID)
	if err != nil {
		return nil, fmt.Errorf("failed to read from world state: %v", err)
	}
	if recordJSON == nil {
		return nil, fmt.Errorf("the record %s does not exist", recordID)
	}

	var record DiagnosticRecord
	err = json.Unmarshal(recordJSON, &record)
	if err != nil {
		return nil, err
	}

	return &record, nil
}

// RecordExists returns true when record with given ID exists in world state
func (s *SmartContract) RecordExists(ctx contractapi.TransactionContextInterface, recordID string) (bool, error) {
	recordJSON, err := ctx.GetStub().GetState(recordID)
	if err != nil {
		return false, fmt.Errorf("failed to read from world state: %v", err)
	}

	return recordJSON != nil, nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		fmt.Printf("Error creating X-Ray smart contract: %s", err.Error())
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting X-Ray smart contract: %s", err.Error())
	}
}
