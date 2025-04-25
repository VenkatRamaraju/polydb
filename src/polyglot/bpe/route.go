package bpe

import (
	"fmt"
)

// Train executes the training process and returns an error if any step in the process fails.
func Train() error {
	// Get data from the source
	pdDataset, err := getData()
	if err != nil {
		return fmt.Errorf("error getting data: %w", err)
	}

	// Notify
	fmt.Println("Done getting data")

	// Perform merges on the statistics
	dataMerges, err := merge(pdDataset)
	if err != nil {
		return fmt.Errorf("error running the BPE algorithm: %w", err)
	}

	// write data to file
	err = WriteMergesMapToJSONFile(dataMerges, "artifacts/merges.json")
	if err != nil {
		return fmt.Errorf("error writing merges map to file: %w", err)
	}

	return nil
}

func GetHigestToken() (int64, error) {
	// Load the merges map
	mapMerges, err := LoadMergesMap()
	if err != nil {
		return -1, fmt.Errorf("Failed to load merges map: %s", err)
	}

	// get the vocab size
	highestToken, err := getHighestToken(mapMerges)
	if err != nil {
		return -1, fmt.Errorf("Error getting vocabulary size: %s", err)
	}

	return highestToken + 1, nil
}
