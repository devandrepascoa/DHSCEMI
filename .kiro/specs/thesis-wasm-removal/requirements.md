# Requirements: Remove WebAssembly References from Thesis

## Overview
Update the thesis to accurately reflect the current implementation, which uses Docker containers and llama.cpp instead of WebAssembly-based solutions.

## Background
The thesis currently describes a WebAssembly-based solution using:
- WebAssembly Component Model
- wasi-nn interfaces
- Dynamic linking with WASI v1
- wRPC for remote module execution
- Cloudflare Workers as a deployment target

The actual implementation is:
- Python FastAPI proxy server
- Docker containers for isolation
- llama.cpp as the inference engine
- Direct CPU/GPU resource management
- OpenAI-compatible API

## User Stories

### 1. Abstract Update
**As a** thesis reader  
**I want** the abstract to accurately describe the implementation approach  
**So that** I understand the actual technical solution used

**Acceptance Criteria:**
- 1.1 Remove all references to WebAssembly, Wasm, Component Model, wasi-nn
- 1.2 Introduce Docker containers as the isolation mechanism
- 1.3 Mention llama.cpp as the inference backend
- 1.4 Remove Cloudflare Workers references
- 1.5 Keep the core problem statement (cost optimization, vertical/horizontal scaling)
- 1.6 Maintain focus on dynamic hardware selection based on cost-benefit analysis

### 2. Portuguese Abstract (Resumo) Update
**As a** Portuguese-speaking reader  
**I want** the Portuguese abstract to match the English abstract  
**So that** both versions are consistent and accurate

**Acceptance Criteria:**
- 2.1 Currently contains placeholder Lorem Ipsum text
- 2.2 Should be translated version of updated English abstract
- 2.3 Should accurately reflect Docker/llama.cpp implementation

### 3. Chapter 4 Architecture Update
**As a** technical reader  
**I want** Chapter 4 to describe the actual architecture  
**So that** I can understand how the system works

**Acceptance Criteria:**
- 3.1 Remove "Remote Module Execution with Dynamic Linking" section
- 3.2 Remove "Remote Module Execution with Component Model and wRPC" section
- 3.3 Add section describing Docker container orchestration
- 3.4 Add section describing llama.cpp integration
- 3.5 Describe FastAPI proxy architecture
- 3.6 Explain container lifecycle management
- 3.7 Describe health checking and readiness probes
- 3.8 Keep cost-benefit analysis and hardware selection logic

### 4. Introduction Chapter Update
**As a** reader  
**I want** the introduction to set correct expectations  
**So that** I'm not confused by WebAssembly references later

**Acceptance Criteria:**
- 4.1 Currently uses template placeholder text
- 4.2 Should introduce the problem domain (cost-effective ML inference)
- 4.3 Should mention containerization as the approach
- 4.4 Should not reference WebAssembly

### 5. Related Work Update
**As a** researcher  
**I want** related work to be relevant to the actual implementation  
**So that** I can understand the context and alternatives

**Acceptance Criteria:**
- 5.1 Remove WebAssembly-specific related work if present
- 5.2 Add related work on container orchestration for ML
- 5.3 Add related work on cost-aware inference serving
- 5.4 Keep DynamoLLM reference as it's relevant
- 5.5 Consider adding references to: Kubernetes autoscaling, Ray Serve, KServe, etc.

### 6. Implementation Chapter Update
**As a** technical reader  
**I want** implementation details to match the codebase  
**So that** I can reproduce or understand the work

**Acceptance Criteria:**
- 6.1 Describe Python/FastAPI implementation
- 6.2 Describe Docker container management
- 6.3 Describe llama.cpp server configuration
- 6.4 Explain metrics collection and decision logic
- 6.5 Show code examples from actual implementation
- 6.6 Remove any WASI SDK, Wasmtime, or Component Model code examples

### 7. Evaluation Chapter Update
**As a** reader  
**I want** evaluation to reflect actual benchmarks  
**So that** I can assess the system's performance

**Acceptance Criteria:**
- 7.1 Use actual benchmark results from benchmarks/ directory
- 7.2 Show CPU vs GPU performance comparisons (RTX 3060 laptop)
- 7.3 Show cost-per-token analysis with actual measured throughputs
- 7.4 Show scaling behavior (cpu_4→cpu_12→gpu_25→gpu_100 transitions)
- 7.5 Reference actual figures from benchmarks/thesis_figures/
- 7.6 Include key observations: CPU sublinear scaling, GPU batch efficiency, gpu_25 latency limitations
- 7.7 Present throughput measurements: cpu_4 (32 tok/s), cpu_12 (47 tok/s), gpu_25 (147 tok/s), gpu_100 (1064 tok/s)
- 7.8 Discuss autoscaling thresholds (80% scale up, 30% scale down)

## Technical Constraints

### Must Preserve
- Core thesis contribution: cost-aware dynamic hardware selection
- Vertical and horizontal scaling concepts
- Cost-benefit analysis approach
- Performance evaluation methodology

### Must Remove
- All WebAssembly/Wasm terminology
- Component Model references
- wasi-nn references
- Dynamic linking discussions
- Cloudflare Workers deployment

### Must Add
- Docker containerization approach
- llama.cpp inference engine
- FastAPI proxy architecture
- OpenAI-compatible API design
- Container lifecycle management

## Out of Scope
- Changing the fundamental thesis contribution
- Rewriting evaluation if benchmarks already exist
- Adding new experiments
- Changing the thesis structure (chapters, sections)

## Dependencies
- Access to actual benchmark results in benchmarks/thesis_figures/
- Understanding of current codebase architecture
- Bibliography updates for new references

## Success Criteria
- No mentions of WebAssembly, Wasm, Component Model, wasi-nn remain
- All architecture descriptions match actual implementation
- Code examples come from actual codebase
- Evaluation uses actual benchmark data
- Thesis reads coherently with new technical approach
