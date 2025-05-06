package main

import (
	"bpe"
	"flag"
	"fmt"
	"log"
	"tgrpc"
)

// main function initializes the application and starts the training process.
func main() {
	// get a function
	psFunction := flag.String("func", "", "Function")
	psFilePath := flag.String("file", "", "Merges file")
	flag.Parse()

	// execute the instruction
	if *psFunction == "t" {
		// train mode
		if err := bpe.Train(); err != nil {
			fmt.Println("Error during training:", err)
		}
	} else if *psFunction == "v" {
		// get vocabulary size
		iVocabSize, err := bpe.GetHigestToken()
		if err != nil {
			fmt.Println("Error while calculating vocabulary size:", err)
			return
		}
		fmt.Println(iVocabSize)
	} else if *psFunction == "e" {
		// get vocabulary size
		err := bpe.EncodeDecode(*psFilePath)
		if err != nil {
			fmt.Println("Error while performing encode and decode:", err)
			return
		}
	} else {
		// open up grpc
		vocab, err := tgrpc.LoadVocab() // your vocab loading logic
		if err != nil {
			fmt.Println("Load vocabulary error:", err)
			return
		}

		// start server
		go func() {
			if err := tgrpc.StartServer(vocab); err != nil {
				log.Fatalf("gRPC server failed: %v", err)
			}
		}()

		// block
		select {}
	}
}
