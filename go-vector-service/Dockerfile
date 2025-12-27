# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod file
COPY go.mod ./

# Initialize go.sum and download dependencies
RUN go mod download || true
RUN go mod tidy

# Copy source code
COPY . .

# Tidy again with all source
RUN go mod tidy

# Build the binary (exclude test files)
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o vector-service ./cmd/server

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates wget

WORKDIR /root/

# Copy the binary from builder
COPY --from=builder /app/vector-service .

# Expose port
EXPOSE 8001

# Set environment variables
ENV VECTOR_SERVICE_PORT=8001
ENV VECTOR_DIMENSIONS=384

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD wget -q --spider http://localhost:8001/health || exit 1

# Run the service
CMD ["./vector-service"]
