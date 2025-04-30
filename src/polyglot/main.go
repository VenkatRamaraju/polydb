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
	psFunction := flag.String("func", "", "Configuration File")
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
			fmt.Println("Error while calculating vocabulary soze:", err)
			return
		}
		fmt.Println(iVocabSize)
	} else {
		// open up grpc
		vocab, err := tgrpc.LoadVocab() // your vocab loading logic
		if err != nil {
			fmt.Println("Load vocabulary error:", err)
			return
		}

		// start server
		go func() {
			if err := tgrpc.Startcd /Users/venkyramaraju/Documents/Studies/Learning/nn/polydb/src/polyvec/proto
			protoc --go_out=. --go_opt=paths=source_relative \
				--go-grpc_out=. --go-grpc_opt=paths=source_relative \
				embeddings.protoServer(vocab); err != nil {
				log.Fatalf("gRPC server failed: %v", err)
			}
		}()

		// block
		select {}
	}
}
