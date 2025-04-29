module polydb

go 1.24.2

require (
	github.com/go-chi/chi/v5 v5.2.1
	go.uber.org/zap v1.27.0
)

require (
	apiserver v0.0.0-00010101000000-000000000000 // indirect
	go.uber.org/multierr v1.10.0 // indirect
)

replace apiserver => ./src/apiserver
