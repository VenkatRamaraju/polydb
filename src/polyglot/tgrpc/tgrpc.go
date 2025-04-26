package tgrpc

import (
	"context"
	"fmt"
	"log"
	"net"

	"bpe"
	pb "proto/tokenizerpb"

	"google.golang.org/grpc"
)

type Server struct {
	pb.UnimplementedTokenizerServer
	vocab map[string]interface{}
}

// Encode implements the Encode RPC
func (s *Server) Encode(ctx context.Context, req *pb.EncodeRequest) (*pb.EncodeResponse, error) {
	convertedMap, ok := s.vocab["merges"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("merges map not found or invalid")
	}

	tokens, err := bpe.Encode(convertedMap, req.Text)
	if err != nil {
		return nil, err
	}

	tokenTexts, err := bpe.ListToTokens(tokens, convertedMap)
	if err != nil {
		return nil, err
	}

	return &pb.EncodeResponse{
		Tokens:     tokens,
		TokenTexts: tokenTexts,
	}, nil
}

// StartServer starts the gRPC server
func StartServer(vocab map[string]interface{}) error {
	// Listen on Unix socket
	lis, err := net.Listen("unix", "/tmp/tokenizer.sock")
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterTokenizerServer(grpcServer, &Server{vocab: vocab})

	log.Println("gRPC server running at /tmp/tokenizer.sock")
	return grpcServer.Serve(lis)
}
