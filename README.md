# Synap-Forge-LLM
![Local photo](./images/image.webp)

[![License: MIT](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://opensource.org/license/apache-2-0)
[![Rust](https://img.shields.io/badge/rust-%23000000.svg?logo=rust&logoColor=white)](https://www.rust-lang.org)

A lightweight, high-performance server that provides OpenAI-compatible endpoints for various Large Language Models (LLMs). This allows you to use any OpenAI-compatible client library or application with different LLM backends.

## Features
- 🚀 Drop-in replacement for OpenAI API endpoints services
- 🔄 Support for multiple LLM backends:
    - [x] Llama 2/3
    - [ ] Mistral
    - [ ] Phi-3
    - [ ] Custom models (extensible architecture)
- ⚡️ Async/Sync processing for high performance
    - [x] Sync processing
    - [ ] Async processing
- 🔑 API key authentication
    - [ ] Token-based Authentication
    - [ ] OAuth/JWT Supports
- 🔍 Detailed logging and monitoring
    - [x] Support different Logging Level (e.g. Debug, Info, Error)
      - `RUST_LOG=info cargo run --features metal`
- 🛡 Error handling and automatic retries

## Quick Start

### Installation

#### MacOS
```bash
cargo build --release --features metal
```

#### Windows/Linux with NVidia GPU
```bash
cargo build --release --features cuda
```

### Basic Usage

#### Prerequisite

```bash
export HF_TOKEN=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
```

1. Start the server:

```bash
./target/synap-forge-llm
```

2. Use with any OpenAI client:

```bash
pip install openai
```

```python
from openai import OpenAI, DefaultHttpxClient

client = OpenAI(
    api_key='EMPTY',
    base_url="http://localhost:8000/v1",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Who won the world series in 2020?"
        },
        {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020."
        },
        {
            "role": "user",
            "content": "Where was it played?"
        }
    ],
    model="gpt-4o",
)

print(chat_completion)
```

## Roadmap
[Roadmap of the project](https://github.com/users/synap-forge/projects/1)

## API Endpoints

The server implements standard OpenAI-compatible endpoints:

- [x] `/v1/chat/completions` - Chat completions API
- [x] `/v1/completions` - Text completions API
- [ ] `/v1/embeddings` - Text embeddings API
- [ ] `/v1/models` - Available models list

## Docker Support
Make sure, your docker platform is supporting [NVidia](https://github.com/NVIDIA/nvidia-container-toolkit) 

```bash
cargo build --release --features cuda
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

- [ ] Request latency
- [ ] Token usage
- [ ] Error rates
- [ ] Model loading time
- [ ] GPU memory usage

## Security Considerations

- [ ] API keys are required by default
- [ ] Rate limiting per API key
- [ ] Input validation and sanitization
- [ ] Configurable maximum token limits
- [ ] Request logging and audit trail

## License

This project is licensed under the Apache License Version 2.0, January 2004 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the API specification
- HuggingFace for model implementations
- Axum for the web framework

## Support

- [Documentation](https://synap-forge-llm.readthedocs.io/)
- [GitHub Issues](https://github.com/synap-forge/synap-forge-llm/issues)
- [Discord Community](https://discord.gg/vxhGShNJ)