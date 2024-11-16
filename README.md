# Synap-Forge-LLM

[![License: MIT](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://opensource.org/license/apache-2-0)
[![Rust](https://img.shields.io/badge/rust-%23000000.svg?logo=rust&logoColor=white)](https://www.rust-lang.org)

A lightweight, high-performance server that provides OpenAI-compatible endpoints for various Large Language Models (LLMs). This allows you to use any OpenAI-compatible client library or application with different LLM backends.

## Features
- 🚀 Drop-in replacement for OpenAI API endpoints
- 🔄 Support for multiple LLM backends:
    - Llama 2/3
    - Mistral
    - Phi-3
    - Custom models (extensible architecture)
- ⚡️ Async processing for high performance
- 🔑 API key authentication
- 📊 Request rate limiting and quota management
- 🔍 Detailed logging and monitoring
- 🛡️ Error handling and automatic retries

## Quick Start

### Installation

```bash
cargo build --release --features metal
```

### Basic Usage

1. Start the server:

```bash
./target/synap-forge-llm
```

2. Use with any OpenAI client:

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="local-llama2",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
```

## API Endpoints

The server implements standard OpenAI-compatible endpoints:

- `/v1/chat/completions` - Chat completions API
- `/v1/completions` - Text completions API
- `/v1/embeddings` - Text embeddings API
- `/v1/models` - Available models list

## Docker Support

```bash
docker build -t synap-forge-llm .
docker run -p 8000:8000 -v llm-proxy-server
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Performance Optimization

### Hardware Acceleration

The server automatically detects and uses available hardware acceleration:

- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- CPU fallback with optimized threading

## Monitoring

Built-in Prometheus metrics will be available at `/metrics`:

- Request latency
- Token usage
- Error rates
- Model loading time
- GPU memory usage

## Security Considerations

- API keys are required by default
- Rate limiting per API key
- Input validation and sanitization
- Configurable maximum token limits
- Request logging and audit trail

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the API specification
- HuggingFace for model implementations
- Axum for the web framework

## Support

- [Documentation](https://llm-proxy-server.readthedocs.io/)
- [GitHub Issues](https://github.com/yourusername/llm-proxy-server/issues)
- [Discord Community](https://discord.gg/)